#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda init bash 
conda activate sra-v2 

for dat in $(ls /home/delvinso/sra-v2/cleaned_data/*tsv )
do
  bn_dat=$(basename ${dat})
  dat_name=$(cut -d'_' -f1 <<< ${bn_dat})
  echo ${dat_name}
  python3 /home/delvinso/sra-v2/embed_papers/preprocess_raw.py \
            --data-path=${dat} \
            --output=/home/delvinso/sra-v2/pickles/raw/${dat_name}.p
done
