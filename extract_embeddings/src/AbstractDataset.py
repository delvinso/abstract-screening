
import torch
import pandas as pd
import numpy as np
from torch.utils import data


class AbstractDataset(data.Dataset):

  def __init__(self, data, list_IDs: list):
    """Create custom torch Dataset.
  
    Arguments:
    data {array-like} --  DataFrame containing dataset.
    list_IDs {list} -- List of data IDs to be loaded.
    labels {dict} -- Map of data IDs to their labels.
    
    Returns:
    X,Y -- data and label.
    """
    self.data = data
    self.list_IDs = list_IDs

  def __len__(self):
    return len(self.list_IDs)

  def __getitem__(self, index):
    # select sample
    ID = self.list_IDs[index] 
    
    X = self.data[self.data['unq_id'] == ID]['All_Text'].values

    # print(self.labels[1:100])
    # y = self.labels[ID] # because this uses a unique index we can no longer take the index.
    # the ith index of the instance
    y = self.data['Inclusion'][self.data['unq_id'] == ID].values[0]
    y2 = self.data['FullText_Inclusion'][self.data['unq_id'] == ID].values[0]
    return ID, torch.tensor(X[0]['input_ids']), torch.tensor(y), torch.tensor(y2)
