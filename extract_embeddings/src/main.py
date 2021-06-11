
import json
import argparse 
import logging
import os
import pandas as pd
import numpy as np
import pickle

# for dealing with multiprocessing/len(ancdata) error
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048*2, rlimit[1]))

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from AbstractBert import AbstractBert 
from AbstractDataset import AbstractDataset


import nltk 
from nltk.corpus import stopwords
import string
import os
import re
import pandas as pd 
import pickle
import numpy as np
from pprint import pprint
from glob import glob

# @Chantal
def clean_text(s):
    """ Basic cleaning for BoW/TFIDF """
    s = s.str.lower()                            
    s = s.str.replace(r'_', ' ')              
    s = s.str.replace(r'\W', ' ')   
    stop = set(stopwords.words('english'))
    s = s.apply(lambda x: [word for word in x.split() if word not in stop])
    s = s.apply(lambda x: [word for word in x if len(word) > 1])
    s = s.apply(lambda x: [word for word in x if not word.isnumeric()])

    return s

def preprocess_df(dataset: pd.DataFrame):
    
       
    # fill with white space so joining doesn't go haywire

    dataset = dataset.fillna(' ')

    #print(dataset.isna().sum())
    dataset = dataset.replace(r'\\r',' ', regex = True)
    dataset = dataset.replace(r'\\t',' ', regex = True)

    dataset['All_Text'] = dataset.agg('{0[Title]} {0[Abstract]}'.format, axis=1)

    dataset[['All_Text_Clean']] = dataset[['All_Text']].apply(lambda x: clean_text(x))
    dataset['All_Text_Clean'] = dataset['All_Text_Clean'].str.join(' ')

    # for lbl in ['Inclusion', 'Final_Inclusion']:
        
    #     if  'Included' in dataset[lbl].values:
    #         dataset[[lbl]] = [0 if i == "Excluded" else 1
    #                                 for i in dataset[lbl]]
    
    
    d = {}
    d['ids'] = np.array(dataset.unq_id)
    try:
        d['labels'] = dataset.Inclusion.values
    except AttributeError:
        d['labels'] = None
    try:
        d['final_labels'] = dataset.FullText_Inclusion.values
    except AttributeError:
        d['final_labels'] = None
    d['title'] = dataset.Title.values
    d['all_text'] = dataset.All_Text.values
    d['all_text_clean'] = dataset.All_Text_Clean.values
    
    return d

#TODO: check for existing embedding before tokenization step - done
#TODO: add final labels if applicable - done

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help = "path to tsv where each row is an article and columns are minimum of the following: unq_id, All_Text, Inclusion")
    parser.add_argument('--config_dir', help='path to the json config file', required = False)
    args = parser.parse_args()

    return parser


def load_data(config, vocab,  max_len: int=None):
    """Load data using PyTorch DataLoader.

    Keyword Arguments:
        config {string} -- config file containing data paths and tokenizer information
        max_len {int} -- maximum token length for a text. (default: {128})
        partition {dict} -- maps lists of training and validation data IDs (default: {None})
        labels {dict} -- (default: {None})

    Returns:
        torch.utils.data.Dataset -- dataset
    """

    unique_id_col = 'unq_id'
    text_col = 'All_Text'
    label_col = 'Inclusion'
    label2_col = 'FullText_Inclusion'

    dataset = pd.read_csv(config['data'], sep='\t')
    print(dataset.head())

    ids = list(dataset[unique_id_col]) # the ids corresponding to each abstract
    total_len = len(ids)


    # set parameters for DataLoader -- num_workers = cores
    #params = {'batch_size': 32, 'shuffle': True}
    #           'shuffle': True,
    #           'num_workers': 0
    #           }

    tokenizer = AutoTokenizer.from_pretrained(vocab)

    dataset[[text_col]] = dataset[text_col].apply(lambda x: tokenizer.encode_plus(str(x), \
                                                                                max_length=config['max_len'], \
                                                                                add_special_tokens=True, \
                                                                                pad_to_max_length=True, \
                                                                                truncation=True))

    if 'FullText_Inclusion' not in dataset.columns: 
        dataset['FullText_Inclusion'] = np.NaN

    # create train/valid generators
    dset = AbstractDataset(data=dataset, list_IDs=ids)

    generator = DataLoader(dset, batch_size = config.get('nn_train_batch_size', 18), shuffle = False) # preserve the order of the IDs, won't matter since it's not being fed into a NN

    return generator

def extract_embeddings(config, vocab, generator):
    """Load embeddings either from cache or from scratch
    Args:
        config (json) -- file configurations.
        name --
       _generator --
    """
    
    data_name =  config['data'].split('/')[-1][:-4] # retrieve file name without the extension
    embed_pkl_f = os.path.join(config['cache'], data_name + '_' + config['embed_type'] + '_embeddings.p')
 
    # get embeddings from scratch
    tokenizer = AutoTokenizer.from_pretrained(vocab)
    embedding_model = AbstractBert(vocab) 

    if torch.cuda.device_count() > 1:
        print("GPUs Available: ", torch.cuda.device_count())
        embedding_model = torch.nn.DataParallel(embedding_model, device_ids=[0, 1, 2])
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    embedding_model.eval().to(device)

    logger.info(' Getting BERT/ROBERTA embeddings...')

    embeddings = _get_bert_embeddings(generator, embedding_model)

    # save embeddings
    pickle.dump(embeddings, open(embed_pkl_f, 'wb'))

    logger.info(' Saved full BERT/ROBERTA embeddings.')

    # embedding_shape = train_embeddings['embeddings'][1].shape[0]

    # return embedding_shape, train_embeddings, valid_embeddings


def _get_bert_embeddings(generator, embedding_model: torch.nn.Module):
    """Get BERT embeddings from a dataloader generator.
    Arguments:
        data_generator {data.Dataset} -- dataloader generator (AbstractDataset).
        embedding_model {torch.nn.Module} -- embedding model. 

    Returns:
        embeddings {dict} -- dictionary containing ids, augmented embeddings, and labels. 
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    #with torch.set_grad_enabled(False):
    with torch.no_grad():
        embeddings = {'ids': [],
                      'embeddings': [],
                      'labels': [],
                      'final_labels': []
                      }

        # get BERT training embeddings

        for local_ids, local_data, local_labels, local_labels2 in generator:
            local_data, local_labels, local_labels2 =  local_data.to(device).long().squeeze(1), \
                                        local_labels.to(device).long(), \
                                        local_labels2.to(device).long()
            #print(local_data[0].shape)
            augmented_embeddings = embedding_model(local_data)

            embeddings['ids'].extend(np.array(local_ids))
            embeddings['embeddings'].extend(np.array(augmented_embeddings.detach().cpu()))
            embeddings['labels'].extend(np.array(local_labels.detach().cpu().tolist()))
            embeddings['final_labels'].extend(np.array(local_labels2.detach().cpu().tolist()))

        #print(embeddings['final_labels'])

        # for k in embeddings:
        #     embeddings[k] = np.array(embeddings[k])
    return embeddings



def _set_logger(config):
    """ set up logging files """

    #log_out = os.path.join(config['out_dir'], name+'_model.log')
    logger = logging.getLogger(__name__)
    #logging.basicConfig(level=logging.INFO, filename = log_out)

    return logger


def _set_dirs(config):
    """ set up directories """
    if config.get('cache') is not None:
        if not os.path.exists(config['cache']): os.makedirs(config['cache'])
        # if not os.path.exists(config['cache']+"/"+name): os.makedirs(config['cache']+"/"+name)

    return 

if __name__ == '__main__':

    args = get_parser().parse_args()
    for arg in vars(args):
        print(arg + ": " + str(getattr(args, arg)))


    if not args.config_dir:
        logger.info('Using command line arguments since json not found.')
    else:
        with open(args.config_dir) as f:
            config = json.load(f)



    logger = _set_logger(config)
    _set_dirs(config)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])


    # check if file already exists
    data_name =  config['data'].split('/')[-1][:-4] # retrieve file name without the extension
    embed_pkl_f = os.path.join(config['cache'], data_name + '_' + config['embed_type'] + '_embeddings.p')



    if os.path.exists(embed_pkl_f):
        raise ValueError("Embeddings already exist...")

    if config.get('embed_type') == 'raw': 
        dataset =  pd.read_csv(config['data'], sep='\t')
        raw_d = preprocess_df(dataset)
        pickle.dump(raw_d, open(embed_pkl_f, 'wb'))


    elif config.get('embed_type') in ['bert', 'roberta']:

        # load bert-related stuff
        bert_models = {'bert':'allenai/scibert_scivocab_uncased',
                    'roberta' : 'allenai/biomed_roberta_base'}

        vocab = bert_models[config['embed_type']]

        gen = load_data(config, vocab)
        extract_embeddings(config, vocab, generator = gen)
