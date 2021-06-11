"""
Description: Performs cross-validation given pre-computed embeddings (tf-idf or transformer), a label
Author: Delvin So

- HPF version submits a job for each embedding (folder) vs each dataset-model-embedding combination
- submits job for each for each of 5 embeddings (11 datasets x 5 models) jobs instead of (11 datasets x 5 models x 5 embeddings) jobs
- comment out for f in glob(args.data +'/*p'): and unindent the corresponding logic for this to work on a pickle wise fashion
"""

import json
import pickle
import os
import time
from glob import glob
import logging 
import argparse
import sys

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import scipy.stats as stats


from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score, roc_curve,\
                            auc, average_precision_score, precision_score, recall_score

np.set_printoptions(threshold = np.inf)

from sklearn.feature_extraction.text import CountVectorizer,  TfidfVectorizer
import nltk 
from nltk.corpus import stopwords

# modified version from check_pickles.py
def load_pickle(f):
    """
    f (str): path to a pickle
    
    loads a pickle and returns a dictionary
    
    """
    embeddings = pickle.load(open(f, "rb" ))

    #TODO: deal with this upstream
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

def create_partition(embeddings, frac = 0.5, rs = None, outcome = 'labels'):
    """
    embeddings (dict)
    
    returns a dictionary of training and validation indices, stratified by the labels
    """

    train_data, test_data = train_test_split(embeddings['ids'],
                                         train_size = frac, 
                                         random_state = rs,
                                         stratify = embeddings[outcome])
    
    partition = {'train': train_data, 'valid': test_data}
    
    logger.info('Training Size: {}'.format((len(partition['train']))))
    logger.info('Validation Size: {}'.format((len(partition['valid']))))
    
    return partition


def split_embeddings(partition, embeddings, embed_type = 'transformers'):
    """
    partition (dict): dictionary of training and validation ids
    
    embeddings (dict): 
    
    returns two dictionarys (train and validation), sliced by the partition
    
    """
    # for k in embeddings:
    #     print(k + ' length: {}'.format(len(embeddings[k])))
    
    ks = {
        'transformers':['ids', 'embeddings', 'labels', #'final_labels',
        'title'],
        'bow': ['ids', 'all_text_clean', 'labels', #'final_labels',
        'title'],
        'tfidf': ['ids', 'all_text_clean', 'labels', #'final_labels', 
        'title']
          }
    
    for k in embeddings:
        embeddings[k] = np.array(embeddings[k])


    train_embeddings = {k: embeddings[k][np.where(np.isin(embeddings['ids'], partition['train']))] for k in ks[embed_type]
                              if k in list(embeddings) }

    
    valid_embeddings = {k: embeddings[k][np.where(np.isin(embeddings['ids'], partition['valid']))] for k in ks[embed_type] 
                             if k in list(embeddings) } 
        
    print(sum(train_embeddings['labels'])/ len(train_embeddings['ids']) * 100) 
    print(sum(valid_embeddings['labels'])/ len(valid_embeddings['ids']) * 100)
    
    return train_embeddings, valid_embeddings

def train_and_eval(train_embeddings, valid_embeddings,
                   classifier, param_grid, n_jobs = 64, cv = 10, outcome = 'labels'):
    
    """
    performs k-fold cross-validation using randomized search given a 
    sklearn estimator and a dictionary of parameters to search
    
    returns a dataframe of fitted and predicted values for the training (cv) and held-out sets, respectively
    
    """
    
    grids = RandomizedSearchCV(classifier, \
                                    param_distributions = param_grid, \
                                    cv = cv,
                                    n_jobs = n_jobs,
                                    scoring = {'average_precision', 'roc_auc'}, \
                                    refit='roc_auc')

    # perform the grid search
    start_time = time.time()
    grids.fit(train_embeddings['embeddings'], train_embeddings[outcome])
    end_time = time.time()

    cv_time = end_time - start_time
    print(' CV Parameter search took {} minutes'.format(cv_time/60)) # seconds

    # take cv results into a dataframe and slice row with best parameters
    cv_res = pd.DataFrame.from_dict(grids.cv_results_)
    cv_best_res = cv_res[cv_res.rank_test_roc_auc == 1]

    logger.info('\tAUROC: {}'.format(cv_best_res[['mean_test_roc_auc']]\
                               .iloc[0]['mean_test_roc_auc']))
    logger.info('\tAUPRC: {}'.format(cv_best_res[['mean_test_average_precision']]\
                               .iloc[0]['mean_test_average_precision']))

    train_df = pd.DataFrame({
        'model_probs': grids.predict_proba(train_embeddings['embeddings'])[:, 1],
        'ground_truth': train_embeddings[outcome],
        'set': 'training_fitted',
        'ids': train_embeddings['ids']
    })

    val_df = pd.DataFrame({
        'model_probs': grids.predict_proba(valid_embeddings['embeddings'])[:, 1],
        'ground_truth': valid_embeddings[outcome],
        'set': 'validation',
        'ids': valid_embeddings['ids']
    })

    c_df = pd.concat([train_df, val_df], axis = 0)
    c_df['cv_time'] = cv_time
    c_df['params'] = str(param_grid) # the parameter grid input
    c_df['best_params'] = str(grids.cv_results_['params'][grids.best_index_])
    return c_df

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help = "path of embeddings (pickles)", required = True)
    parser.add_argument('--name', help = "name of experiment", required = True)
    parser.add_argument('--cv', type = int, help = 'number of cv folds', default = 10)
    parser.add_argument('--n_jobs', type = int, help = 'number of threads to use', default = 16)
    parser.add_argument('--outcome', type = str, help = 'abstract inclusion (label) or final inclusion (final_label)', required = True)
    parser.add_argument('--outdir', type = str, help = 'directory to store results')
    parser.add_argument('--num_iters', type = int, help = 'number of iterations to run', default = 1)
    parser.add_argument('--model_type', type = str, help = 'model type, one of lasso, svm, ridge, or rf', required = True)
    parser.add_argument('--embed_type', type = str, help = 'one of tf-idf, bow, or transformers', required = True)
    args = parser.parse_args()
    return parser

if __name__ == '__main__':

    # if not args.config_dir:
    #     logger.info('Using command line arguments since json not found.')
    # else:   
    #     with open(args.config_dir) as f:
    #         config = json.load(f)

    args = get_parser().parse_args()
    for arg in vars(args):
        print(arg + ": " + str(getattr(args, arg)))

    # make dirs
    res_dir = os.path.join(args.outdir)
    if not os.path.exists(res_dir): os.makedirs(res_dir)   

    # set up logger
    log_out = os.path.join(args.outdir, args.name + '.log')
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, filename = log_out)

    # data folder
    pickle_dir = args.data# where pickles are stored
    logging.info(pickle_dir)
    logging.info(args.name)

    # settings
    n_jobs = args.n_jobs
    num_iter = args.num_iters
    cv = args.cv # some datasets don't have enough positive values for a 10 fold cv lol
    outcome = args.outcome #'labels'

    # seed stuff so folds are reproducible and consistent within iterations
    np.random.seed(2020)
    random_seeds = np.random.randint(1e4, size = 100)
    print(random_seeds)

    # may have to change these to the same values here for benchmark + re-sampling
    probabilistic_classifiers = {'knn' : (KNeighborsClassifier(n_jobs=n_jobs),
                                         {'n_neighbors': [3, 5, 7, 11, 13, 15], 'weights': ['distance']}),

                                'svm' : (SVC( random_state = 0, verbose = 1, kernel = 'rbf', probability = True, tol = 0.001),
                                        {'C':  stats.loguniform(1e-4, 1e2)  }), # cap at 1

                                'lasso': (LogisticRegression(n_jobs=n_jobs, penalty='l1', solver = 'saga',
                                                            random_state=0, verbose=1, max_iter=5000, tol=0.001),
                                        {'C': stats.loguniform(1e-4, 1e2) }),

                                'enet': (LogisticRegression(n_jobs=n_jobs, penalty='elasticnet', solver = 'saga',
                                                            random_state=0, verbose=1, max_iter=5000, tol=0.001),
                                        {'C': stats.loguniform(1e-4, 1e2),
                                        "l1_ratio":  stats.uniform(0, 1)}),

                                'ridge': (LogisticRegression(n_jobs=n_jobs, penalty='l2', solver = 'saga',
                                                            random_state=0, verbose=1, max_iter=5000, tol=0.001),
                                        {'C': stats.loguniform(1e-4, 1e2)}),

                                'rf': (RandomForestClassifier(n_jobs=n_jobs, verbose=1, random_state=0),
                                        {'n_estimators': [500, 750, 1000],
                                        'max_features': ["sqrt", 0.05, 0.1],
                                        'min_samples_split': [2, 4, 8]})}


    #f = args.data
    for f in glob(args.data +'/*p'):
        print(f)
        logger.info(f)
        # generate filename from basename
        basen = os.path.basename(f)
        txt = os.path.splitext(basen)[0]
        txt = txt.split('_')[0].replace(' ', '_')
        
        
        print('Dataset: {}'.format(txt))
                    
        # load in embeddings

        f_embeddings = load_pickle(f)

        if outcome not in list(f_embeddings):
            logger.info("{} not found in dataset: {} but specified as outcome".format(outcome, txt))
            sys.exit()
        else:
            print('Label found')

        for i in range(num_iter):
            print(i)
            print(random_seeds[i])
            
            for model_type in [args.model_type]:

                f_partition = create_partition(f_embeddings, rs = random_seeds[i], frac = 0.7)
                classifier, param_grid = probabilistic_classifiers[model_type]
                split1, split2 = split_embeddings(f_partition, f_embeddings,    embed_type = args.embed_type)

                
                if args.embed_type == 'tfidf':

                    tfidfconverter = TfidfVectorizer(min_df=5, max_df=0.7, max_features=1000, stop_words=stopwords.words('english'))
                    tr = tfidfconverter.fit(split1['all_text_clean'])
                    split1['embeddings'] = tr.transform(split1['all_text_clean'])
                    split2['embeddings'] = tr.transform(split2['all_text_clean'])

                elif args.embed_type == 'bow':
                    
                    # https://stackoverflow.com/questions/22489264/is-a-countvectorizer-the-same-as-tfidfvectorizer-with-use-idf-false
                    tfidfconverter = TfidfVectorizer(min_df=5, max_df=0.7, max_features=1000, stop_words=stopwords.words('english'), use_idf = False, norm = None)
                    tr = tfidfconverter.fit(split1['all_text_clean'])
                    split1['embeddings'] = tr.transform(split1['all_text_clean'])
                    split2['embeddings'] = tr.transform(split2['all_text_clean'])

                c_df = train_and_eval(train_embeddings = split1, valid_embeddings = split2,
                                    classifier = classifier, 
                                    param_grid = param_grid, 
                                    cv = cv, 
                                    n_jobs = args.n_jobs,
                                    outcome = outcome)

                c_df['dataset'] = txt
                c_df['method'] = args.name
                #c_df['exchange'] = exchange
                c_df['outcome'] = args.outcome
                c_df['model'] = args.model_type
                c_df['iter'] = i
                c_df.to_csv(os.path.join(res_dir, '{}_{}_false_{}_preds{}.csv'\
                            .format(txt, args.name, args.model_type , i)))
