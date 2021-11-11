
## Systematic Review Accelerator Experiments

Delvin So

This repo is for experiments related to abstract screening for our manuscript entitled *Exploration of automated abstract screening using deep learning embeddings*, where the goal is to compare different deep-learning embeddings (specifically the CLS token) and different classifiers in predicting the probability of abstract inclusion.  Much of the embedding code has been re-factored to account for SPECTER being added into HuggingFace. In the future, if time permits, `run_models.py` could be re-factored to accept jsonl files directly instead of the current pickled dictionary set-up. 

## Preliminary Set-up 

### Directory

After git cloning this repo, your directory should look like:

```
├── data_preprocessing
|   ├── data 
|       ├── datasets_complete                <- place clean datasets here with format below
├── embed_papers
├── environment.yml
├── model_embedding_experiments
├── pickles
├── README.md
├── sample_prevalence_experiments
└── scripts

```

### Conda Environment

Create the `sra-v2` conda environment by running `conda env create -f environment.yml` 

### Embeddings 

HPF GPUs do not have internet access so you will need to download pre-trained [scibert](https://huggingface.co/allenai/scibert_scivocab_uncased),[biomed roberta](https://huggingface.co/allenai/biomed_roberta_base) and [SPECTER](https://huggingface.co/allenai/specter). Alternatively, a local cache can be created by running the code below on the HPF. Replace `cache_dir` with your directory of choice. This directory should be appropriately set in the `Model` and `AbstractDataset` classes in `embed_papers.py`.

```python
# pip install --upgrade transformers
from transformers import AutoTokenizer, AutoModel
import os

models =["allenai/scibert_scivocab_uncased", 'allenai/specter', "allenai/biomed_roberta_base"]
cache_dir = "/home/delvinso/hf-custom-cache"

if not os.path.exists(cache_dir): os.makedirs(cache_dir)

for model_name in models: 
  tokenizer=AutoTokenizer.from_pretrained(model_name)
  tokenizer.save_pretrained(os.path.join(cache_dir, model_name))
  model=AutoModel.from_pretrained(model_name)
  model.save_pretrained(os.path.join(cache_dir, model_name))

# testing - should not have to download anything

for model_name in models: 
  tokenizer=AutoTokenizer.from_pretrained(os.path.join(cache_dir, model_name))
  model=AutoModel.from_pretrained(os.path.join(cache_dir, model_name))

```

## Workflow

Abstract Datasets should be placed in `data_preprocessing/data/datasets_complete/` with the following format:

| Covidence.. | Title                                | Authors      | Abstract | Published.Year | Published.Month | Journal                                  | Volume | Issue | Pages   | Accession.Number                                                                                                                       | DOI | Ref  | Study       | Notes                                 | Tags | Inclusion | FullText_Inclusion |
| ----------- | ------------------------------------ | ------------ | -------- | -------------- | --------------- | ---------------------------------------- | ------ | ----- | ------- | -------------------------------------------------------------------------------------------------------------------------------------- | --- | ---- | ----------- | ------------------------------------- | ---- | --------- | ------------------ |
| #1581       | Voluntary work with Tibetan refugees | "Heslop, P." |          | 1990           |                 | Midwife Health Visitor & Community Nurse | 26     | 4     | 136-140 | 107520452. Language: English. Entry Date: 19900801. Revision Date: 20150712. Publication Type: Journal Article. Journal Subset: Europe |     | 6535 | Heslop 1990 | Exclusion reason: Wrong study design; | SRH  | 1         | 0                  |

A sample dataset, `sample_wash_data.csv`, with 200 abstracts is provided. 

1. to pre-process datasets for consistency, 
```
python3 data_preprocessing/clean_normalize_datasets.py \
    --input-dir=data_preprocessing/data/datasets_complete/ \
    --output-dir=cleaned_data
```
    -  input: `data_preprocessing/data/datasets_complete`
    -  output: `cleaned_data/`


2. a) run `embed_papers/embed_papers.sh`, `python3 embed_papers/jsonl_to_pickles.py`  to embed datasets, outputting json, and convert it to a pickled dictionary. Note; we could technically avoid creating the dictionary/pickles if `run_model.py` directly accepted jsonl but in the interest of time it is easier to convert the jsonl into a format already accepted by the script. 

    - `embed_papers.sh` should be run with access to GPUs, on HPF eg. `qsub embed_papers/embed_papers.sh -q gpu -l nodes=1:ppn=1:gpus=1,vmem=120g,mem=80g,walltime=6:00:00` 
    - input: `cleaned_data/`
    - output: `embed_papers/{bert|roberta|specter}`, `pickles/{bert|roberta|specter}`

   b) run `embed_papers/preprocess_raw.sh`
    - input: `cleaned_data/`
    - output: `pickles/raw`
    
3. bash scripts stored in `model_embedding_experiments/` which run all dataset-embedding-model combinations in the manuscript. 
    - locally (not on HPC): `./run_frequency_embeddings.sh && ./run_frequency_embeddings.sh` - NOT TESTED IN V2
    - on hpf: `./submit_transformers_all.sh && ./submit_freq_all.sh`
    - input: `pickles/{bert|roberta|specter|raw}`
    - output: `results/threads`

