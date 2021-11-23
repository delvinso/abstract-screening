
import json
import argparse 
import os
import pandas as pd
import pickle
import nltk 
from nltk.corpus import stopwords
import string
import re
import numpy as np

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
    dataset = dataset.replace(r'\\r',' ', regex = True)
    dataset = dataset.replace(r'\\t',' ', regex = True)

    dataset['All_Text'] = dataset.agg('{0[Title]} {0[Abstract]}'.format, axis=1)
    dataset[['All_Text_Clean']] = dataset[['All_Text']].apply(lambda x: clean_text(x))
    dataset['All_Text_Clean'] = dataset['All_Text_Clean'].str.join(' ')

    d = {}
    d['ids'] = np.array(dataset.unq_id)

    d['labels'] = dataset.Inclusion.values

    try:
        d['final_labels'] = dataset.FullText_Inclusion.values
    except AttributeError:
        d['final_labels'] = None
    d['title'] = dataset.Title.values
    d['all_text'] = dataset.All_Text.values
    d['all_text_clean'] = dataset.All_Text_Clean.values
    
    return d


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', help = "path to tsv where each row is an article and columns are minimum of the following: unq_id, All_Text, Inclusion", required = True)
    parser.add_argument('--output', help = "path to write the output json.")
    args = parser.parse_args()
    return parser


if __name__ == '__main__':

    args = get_parser().parse_args()
    for arg in vars(args):
        print(arg + ": " + str(getattr(args, arg)))
    
    if not os.path.exists(os.path.dirname(args.output)): os.makedirs(os.path.dirname(args.output))

    # check if file already exists
    data_name =  args.data_path.split('/')[-1][:-4] # retrieve file name without the extension


    if os.path.exists(args.output):
        raise ValueError("Embeddings already exist...")

    dataset =  pd.read_csv(args.data_path, sep='\t')
    raw_d = preprocess_df(dataset)
    pickle.dump(raw_d, open(args.output, 'wb'))

