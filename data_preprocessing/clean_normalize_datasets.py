"""
Delvin so and Chantal Shaib

Code could definitely be cleaned up as it has been ported from a jupyter notebook.

python3 data_preprocessing/clean_normalize_datasets.py \
    --input-dir=data_preprocessing/data/datasets_complete/ \
    --output-dir=cleaned_data
"""

import os
import warnings
import pandas as pd
import argparse 
import re 
import csv
from glob import glob
import ssl
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

warnings.filterwarnings('ignore')

def preprocess_df(dataset: pd.DataFrame) -> pd.DataFrame:
    
    print('\tBefore ' + str(dataset.shape[0]))

    # we want to drop on empty abstracts, but not titles
    
    dataset['unq_id'] = dataset.index.values.tolist()
    
    dataset['Inclusion'] = pd.to_numeric(dataset['Inclusion'], errors = 'coerce' )
    
    dataset = dataset[~dataset['Abstract'].isnull()]
    dataset = dataset[~dataset['Inclusion'].isnull()]

    # fill with white space so joining doesn't go haywire

    dataset = dataset.fillna(' ')

    dataset = dataset.replace(r'\\r',' ', regex = True)
    dataset = dataset.replace(r'\\t',' ', regex = True)

    dataset['All_Text'] = dataset.agg('{0[Title]} {0[Abstract]}'.format, axis=1)
    
    dataset['Metadata'] = dataset.agg('{0[Authors]} {0[Published.Year]} {0[Journal]} {0[Notes]}'.format, axis=1)
    
    print('\tAfter ' + str(dataset.shape[0]))
    
    return dataset


# TODO: Do we want to remove numbers and special characters (e.g., other languages??)
# Credit to Chantal
def clean_text(s:str) -> str:
    s = s.str.lower()                         # put to lowercase for homogeneity    
    s = s.str.replace(r'_', ' ')              # remove underscores from the notes
    s = s.str.replace(r'\W', ' ')             # remove punctutation
    stop = set(stopwords.words('english'))    # define stop words
    lemmatizer = WordNetLemmatizer()          # lemmatize - a lot of repeat words
    s = s.apply(lambda x: [lemmatizer.lemmatize(word, 'v')
                              for word in x.split() 
                              if word not in stop]) # remove stopwords

    s = s.apply(lambda x: [word for word in x if len(word) > 1])
    s = s.apply(lambda x: [word for word in x if not word.isnumeric()])

    return s


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type = str, required= True, help='path to folder of csvs containing abstract metadata')
    parser.add_argument('--output-dir', type = str, required = True, help='path to write the processed data to.')
    parser.add_argument('--check-output-exists', type = int, default = 0, help='check if files already exist, exit if so')

    return parser


if __name__ == "__main__":
    
    args = get_parser().parse_args()
    for arg in vars(args):
        print(arg + ": " + str(getattr(args, arg)))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # need to only download only once
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    # collect the datasets into a dict
    reviews = {}
    
    # see if output already exists, otherwise set to 0
    check = args.check_output_exists
 
    # TODO: to be consistent with the rest of the code, the script should take a single file and process it 
    for f in glob(os.path.join(args.input_dir, '*')):

        print(f)

        key = re.split(r'_', os.path.basename(f))[0]

        out_fn = os.path.join(args.output_dir, key + '_oct.tsv')

        if check == 1 and os.path.exists(out_fn): 
            print(f'Output file already exists for {key}, skipping!')
            continue

        else:
            print(f'Reading in {f}....')
            if f.endswith('csv'):
                reviews[key] = pd.read_csv(f, encoding='latin1')
            elif f.endswith('xlsx'):
                reviews[key] = pd.read_excel(f)


    # normalize col names where necessary 

    to_keep = ['Title', 'Abstract', 'Notes','Published.Year', 
                'Covidence..', 'Inclusion', 'FullText_Inclusion',
                'Authors', 'Journal']

    for key in reviews:

        dataset = reviews[key]
        
        if 'Published Year' in dataset.columns.tolist():
            dataset.rename(columns = {'Published Year':'Published.Year'}, 
                    inplace=True)
            
        if 'Covidence #' in dataset.columns.tolist():
            dataset.rename(columns = {'Covidence #':'Covidence..'}, 
                    inplace=True)
            
        filter_col = [col for col in dataset if col in to_keep]

        dataset = dataset[filter_col].copy()
    
        # preprocess the data

        dataset = preprocess_df(dataset)

        print(key)
        dataset[['All_Text_Clean']] = dataset[['All_Text']].apply(lambda x: clean_text(x))
        dataset['All_Text_Clean'] = dataset['All_Text_Clean'].str.join(' ')
        
        dataset[['Metadata_Clean']] = dataset[['Metadata']].apply(lambda x: clean_text(x))
        dataset['Metadata_Clean'] = dataset['Metadata_Clean'].str.join(' ')

        # save down the data
        out_fn = os.path.join(args.output_dir, key + '_oct.tsv')
        
        #if not os.path.isfile(fn):
        if 'FullText_Inclusion' not in dataset.columns.tolist():
            dataset[['unq_id', 'All_Text_Clean', 'Metadata_Clean', 'Inclusion','Covidence..',
                    'Title', 'Abstract', 'All_Text', 'Metadata']].to_csv(out_fn, index = False, sep = '\t', quoting=csv.QUOTE_NONNUMERIC)
        else: 
            dataset[['unq_id', 'All_Text_Clean', 'Metadata_Clean', 'Inclusion', 'FullText_Inclusion', 'Covidence..',
                    'Title', 'Abstract', 'All_Text', 'Metadata']].to_csv(out_fn, index = False, sep = '\t', quoting=csv.QUOTE_NONNUMERIC)

        print('{} successfully saved!'.format(out_fn))