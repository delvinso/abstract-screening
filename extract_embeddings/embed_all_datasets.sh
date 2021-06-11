#!/bin/bash
# conda activate abstract
mkdir configs
for dat in $(ls ../cleaned_data/*tsv )
do
  bn_dat=$(basename ${dat})
  dat_name=$(cut -d'_' -f1 <<< ${bn_dat})
  echo ${dat_name}
  for embed_type in   'bert' 'roberta' 'raw'
  do
            echo "
            {
              \"data\" : \"${dat}\",
              \"max_len\" : 512,
              \"seed\" : 2020,
              \"cache\": \"../pickles\/${embed_type}\",
              \"embed_type\":\"${embed_type}\",
              \"nn_batch_size\": \"18\"
            }" > configs/config_sample.json

            #cat configs/config_sample.json

            python src/main.py --config_dir=configs/config_sample.json 
  done
done



