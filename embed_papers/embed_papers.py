import pandas as pd
import os 
from transformers import AutoModel, AutoTokenizer
import json
import argparse
from tqdm.auto import tqdm
import pathlib
import torch
import numpy as np


class AbstractDataset:

    def __init__(self, data_path, model_name, cache_dir='', max_length=512, batch_size=32):
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.cache_dir, self.model_name))
        self.max_length = max_length
        self.batch_size = batch_size
        self.data = pd.read_csv(data_path, sep='\t').to_dict(orient = 'records')


    def __len__(self):
        return len(self.data)

    def batches(self):
        # create batches, credits to # https://github.com/allenai/specter/blob/master/scripts/embed_papers_hf.py
        batch = []
        batch_ids = [] # paper identifiers
        labels = []    # abstract inclusion
        labels2 = []   # final paper inclusion
        
        
        batch_size = self.batch_size
        
        
        for i, d in enumerate(self.data):
            
            if (i) % batch_size != 0 or i == 0:
                batch_ids.append(d['unq_id'])
                batch.append(d['Title'] + ' ' + (d.get('Abstract') or ''))
                labels.append(d['Inclusion'])
                labels2.append(d.get('FullText_Inclusion', np.nan))
                
            else:
                input_ids = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)
                yield input_ids.to('cuda'), batch_ids, labels, labels2
                batch_ids = [d['unq_id']]
                labels = [d['Inclusion']]
                labels2 = [d.get('FullText_Inclusion', np.nan)]
                batch = [d['Title'] + ' ' + (d.get('Abstract') or '')]
                
                
        if len(batch) > 0:
            input_ids = self.tokenizer(batch, padding=True, truncation=True, 
                                       return_tensors="pt", max_length=self.max_length)        
            input_ids = input_ids.to('cuda')
            yield input_ids, batch_ids, labels, labels2

class Model:

    def __init__(self, model_name, cache_dir):
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(os.path.join(self.cache_dir, self.model_name))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def __call__(self, input_ids):
        output = self.model(**input_ids)
        return output.last_hidden_state[:, 0, :] # cls token

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', help='path to a csv containing abstract metadata')
    parser.add_argument('--output', help='path to write the output json.')
    parser.add_argument('--model-name', type=str, help='model name as found on HuggingFace, eg. allenai/specter')
    parser.add_argument('--batch-size', type=int, default=8, help='batch size for inference')
    return parser


def main():
    args = get_parser().parse_args()
    for arg in vars(args):
        print(arg + ": " + str(getattr(args, arg)))

    cache_dir = "/home/delvinso/hf-custom-cache"
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    dataset = AbstractDataset(cache_dir=cache_dir, data_path=args.data_path, batch_size=args.batch_size, model_name= args.model_name)
    model = Model(cache_dir=cache_dir, model_name= args.model_name)
    results = {}
    batches = []
    for batch, batch_ids, labels, labels2 in tqdm(dataset.batches(), total=len(dataset) // args.batch_size):
        batches.append(batch)
        emb = model(batch)
        for paper_id, label, label2, embedding in zip(batch_ids, labels, labels2, emb.unbind()):
            results[paper_id] =  {"paper_id": paper_id, 
                                  "label" : label,
                                  "final_label" : label2,
                                  "embedding": embedding.detach().cpu().numpy().tolist()}


    
    with open(args.output, 'w') as fout:
        for res in results.values():
            fout.write(json.dumps(res) + '\n')

if __name__ == '__main__':
    main()
    
