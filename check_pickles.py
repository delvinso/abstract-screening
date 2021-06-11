"""

Description: Prints out number of abstracts, embedding size, # of labels and # of inclusions given a folder of pickled transformer embeddings
Author: Delvin So

eg. 

# python3 check_pickles.py --folder /home/dso/Documents/Projects/abstract_tool/_model_v2/extract_embeddings/pickles
# python3 check_pickles.py --folder /home/dso/Documents/Projects/abstract_tool/_model_v2/specter_cache
# python3 check_pickles.py --folder /home/dso/Documents/Projects/specter_v2/specter_pickles

# python3 check_pickles.py --folder /home/dso/Documents/Projects/abstract_tool/_model_v2/benchmark/pickles_ds # no compatability with text -> pickles
# python3 check_pickles.py --folder /home/dso/Documents/Projects/abstract_tool/_model_v2/benchmark/specter/pickles/cohen_2006


python3 check_pickles.py --folder /home/dso/Documents/Projects/abstract_tool/_model_v2/extract_embeddings/pickles/specter # 
python3 check_pickles.py --folder /home/dso/Documents/Projects/abstract_tool/_model_v2/extract_embeddings/pickles/bert
python3 check_pickles.py --folder /home/dso/Documents/Projects/abstract_tool/_model_v2/extract_embeddings/pickles/roberta
python3 check_pickles.py --folder /home/dso/Documents/Projects/abstract_tool/_model_v2/extract_embeddings/pickles/raw
"""


import pickle
import argparse 
import numpy as np
from glob import glob 
import os 

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', help = "path of embeddings (pickles) to get statistics on ", required = True)
    args = parser.parse_args()

    return parser

def load_pickle(f):
    """
    f (str): path to a pickle
    
    loads a pickle and returns a dictionary
    
    """
    embeddings = pickle.load(open(f, "rb" ))

    #embeddings['labels'] = embeddings['labels'].ravel()
    
    if 'Included' in embeddings['labels']:
        embeddings['labels'] = [0 if i == "Excluded" else 1 
         for i in embeddings['labels']]
        
    elif embeddings.get('final_labels') is None:
        print('No final inclusion labels')
    elif  'Included' in embeddings.get('final_labels'):
        embeddings['final_labels'] = [0 if i == "Excluded" else 1 
         for i in embeddings['final_labels']]
        embeddings['final_labels'] = np.array(embeddings['final_labels'])
        
    return embeddings

if __name__ == '__main__':
    
    args = get_parser().parse_args()
    for arg in vars(args):
        print(arg + ": " + str(getattr(args, arg)))


    pickles = glob(os.path.join(args.folder, '*.p*'))

    if len(pickles) == 0:
        print('No pickles found in: {}'.format(args.folder))
    else:
        print(f'{str(len(pickles))} found in {args.folder}')

    for f in pickles:
        print(f'Loading {os.path.basename(f)}')
        f_embeddings = load_pickle(f)


        if isinstance(f_embeddings['labels'] , np.ndarray): # from specter/tf-idf preprocessing
            pos = np.sum(f_embeddings['labels'] == 1)
        elif isinstance(f_embeddings['labels'] , list): # from BERT/RoBERTa embeddings
            pos = f_embeddings['labels'].count(1)
        prev = round(pos/len(f_embeddings['labels']) * 100, 2) 

        print('\t' + '# of Abstracts: ' + str(len(f_embeddings['labels'])))
        try:
            print('\t' + 'Embedding Size: ' + str(f_embeddings['embeddings'][0].shape))
        except KeyError:
            print('\tNo embeddings available')
        print('\t' + '# of Labels:  ' + str(len(f_embeddings['labels'])))
        print('\t' + '# of Inclusions:  ' + str(pos))
        print('\t' + 'Prevalence (%):   ' + str(prev))
        


