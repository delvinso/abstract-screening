## Systematic Review Accelerator Experiments

Delvin So

This repo is for experiments related to abstract screening, where the goal is to compare different deep-learning embeddings (specifically the CLS token) and different classifiers in predicting the probability of abstract inclusion. 

## Set-up Conda Environments

For anything unrelated to specter:
`conda env create -f environment.yml`

This study was conducted prior to specter being released on hugging face.

For specter embedding, follow the steps found here:
https://github.com/allenai/specter

## Directory Set-Up

After git cloning this repo, your directory should look like:

```
.
|-- README.md
|-- check_pickles.py
|-- cleaned_data                             
|-- data_preprocessing
|   |-- notebooks
|   |-- data 
|       |-- datasets_complete                <- place clean datasets here with format below
|-- environment.yml
|-- extract_embeddings
|-- logs
|-- pickles
|-- results
|-- run_models.py
|-- scripts
|-- submit_freq_all.sh
`-- submit_transformers_all.sh

```

## Workflow

Abstract Datasets should be placed in `data_preprocessing/data/datasets_complete/` with the following format:

Covidence..|Title|Authors|Abstract|Published.Year|Published.Month|Journal|Volume|Issue|Pages|Accession.Number|DOI|Ref|Study|Notes|Tags|Inclusion|FullText_Inclusion
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |--- |---
#1581|Voluntary work with Tibetan refugees|"Heslop, P."||1990||Midwife Health Visitor & Community Nurse|26|4|136-140|107520452. Language: English. Entry Date: 19900801. Revision Date: 20150712. Publication Type: Journal Article. Journal Subset: Europe||6535|Heslop 1990|Exclusion reason: Wrong study design; |SRH|1|0

A sample dataset, `sample_wash_data`, can be found with 200 abstracts.


**There is currently no support for steps 1-3 being run through SGE but they should run fine on an interactive node with a GPU. Furthermore steps 1-3 assume you have access to a GPU and can take hours to run depending on the size of the dataset!!**

on HPF GPUs do not have internet access so make sure to download scibert and roberta from https://huggingface.co/allenai/scibert_scivocab_uncased and https://huggingface.co/allenai/biomed_roberta_base respectively before extracting the embeddings.

Make sure to change input and output directories, notably in the specter embedding notebook and when running the experiments (`submit*_all.sh`)!

1. run `data_preprocessing/notebooks/data_clean_exploration_v2.ipynb` to pre-process datasets
    -  input: `data_preprocessing/data/datasets_complete`
    -  output: `cleaned_data/`
2. run `data_preprocessing/notebooks/embed_specter_sar_v2.ipynb`
    - input: `cleaned_data/`
    - output:
        - `data_preprocessing/data/specter`
        - `pickles/specter`
3. run `embed_all_datasets.sh`
    - input: `cleaned_data/`
    - output:
        - `pickles/raw`, `pickles/bert`, `pickles/roberta`
4. run experiments. bash scripts which run all dataset-embedding-model combinations in the manuscript. essentially a wrapper around `run_models.py`.
    - locally (not on HPC): `./run_frequency_embeddings.sh && ./run_frequency_embeddings.sh`
    - on torque: `./submit_transformers_all.sh && ./submit_freq_all.sh`
    - input: `pickles/*`
    - output: `results/threads`
    
After running the commands, the output predictions will be in `results/` (or whatever specified directory in submit_*.sh). The directory should look like :

```
|-- README.md
|-- check_pickles.py
|-- cleaned_data
|-- data_preprocessing
|   |-- data
|   |-- notebooks
|   `-- specter
|-- environment.yml
|-- extract_embeddings
|   |-- README.md
|   |-- configs
|   |-- embed_all_datasets.sh
|   `-- src
|-- logs
|-- pickles
|   |-- bert
|   |-- raw
|   |-- roberta
|   `-- specter
|-- results
|-- run_models.py
|-- scripts
|-- submit_freq_all.sh
`-- submit_transformers_all.sh
```

