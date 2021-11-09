#!/bin/bash

# python3 embed_papers.py --data-path=cleaned_data/WASH_oct.tsv --model-name="allenai/specter" --output=WASH.json --batch-size=8

# https://stackoverflow.com/questions/55507519/python-activate-conda-env-through-shell-script
source ~/anaconda3/etc/profile.d/conda.sh
conda init bash 
conda activate sra-v2 

for dat in $(ls /home/delvinso/sra-v2/cleaned_data/*tsv )
do
  bn_dat=$(basename ${dat})
  dat_name=$(cut -d'_' -f1 <<< ${bn_dat})
  echo ${dat_name}
  for embed_model in  'allenai/scibert_scivocab_uncased' 'allenai/specter' 'allenai/biomed_roberta_base'
  do
    echo ${embed_model}
    if [ ${embed_model} == "allenai/scibert_scivocab_uncased" ]; then 
      folder="bert"
    elif [ ${embed_model} == "allenai/specter" ]; then 
      folder="specter"
    elif [ ${embed_model} == "allenai/biomed_roberta_base" ]; then 
      folder="roberta"
    fi
    # could qsub each model here but HPF has limited GPUs
    HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        python3 /home/delvinso/sra-v2/embed_papers/embed_papers.py \
            --data-path=${dat} \
            --model-name=${embed_model} \
            --output=/home/delvinso/sra-v2/embed_papers/${folder}/${dat_name}.json \
            --batch-size=8
  done
done
