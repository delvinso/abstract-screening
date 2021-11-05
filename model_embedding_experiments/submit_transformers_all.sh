#!/bin/bash

out_dir=/home/delvinso/sra-v2/model_embedding_experiments/results/threads
log_dir=/home/delvinso/sra-v2/model_embedding_experiments/results/logs/
mkdir -p $out_dir
mkdir -p $log_dir

for dat in $(ls /home/delvinso/sra-v2/pickles/ | egrep -v raw) # each folder (embedding)
do
  embed=$(basename ${dat})
  #dat_name=$(cut -d'_' -f1 <<< ${bn_dat})
  echo ${dat}
  echo ${embed}
  for model in 'knn' 'lasso' 'enet' 'ridge' 'rf' 
  do
    if [ ${model} == "rf" ]; then 
      wt='24:00:00';threads='16';m='64g';vm='100g' # default 36hrs, 8 threads, 64gb, 100gb
    elif [ ${model} == "knn" ];  then 
      wt='02:00:00';threads='16';m='100g';vm='100g'   # default 2hrs, 8 threads, 64gb, 64gb # UPDATE MEMORY FOR KNN MANUSCRIPT
    else 
      wt='02:00:00';threads='16';m='64g';vm='100g'   # default 2hrs, 8 threads, 64gb, 64gb # UPDATE MEMORY FOR KNN MANUSCRIPT
    fi
    echo ${model}
    qsub <<EOF
#!/bin/bash
#PBS -N ${embed}_${model}
#PBS -l walltime=${wt}
#PBS -l nodes=1:ppn=${threads}
#PBS -l mem=${m},vmem=${vm}
#PBS -o ${log_dir}/${embed}_${model}.e
#PBS -e ${log_dir}/${embed}_${model}.o

source ~/anaconda3/etc/profile.d/conda.sh
conda init bash 
conda activate sra-v2 
cd /home/delvinso/sra-v2

python3 model_embedding_experiments/run_models.py \
  --data=pickles/${dat}  \
  --outcome=labels \
  --name=${embed} \
  --model_type=${model}\
  --outdir=${out_dir} \
  --n_jobs=${threads} \
  --embed_type=transformers
EOF
  done
done