#!/bin/bash
# 15 combinations

out_dir=YOURDIRHERE
log_dir=YOURDIRHERE
mkdir -p $out_dir
mkdir -p $log_dir


for dat in $(ls YOURDIRHERE| egrep -v raw) # each folder (embedding)
do
  embed=$(basename ${dat})
  #dat_name=$(cut -d'_' -f1 <<< ${bn_dat})
  echo ${dat}
  echo ${embed}
  for model in 'lasso' 'knn' 'enet' 'ridge' 'rf'
  do
    if [ ${model} == "rf" ]; then 
      wt='24:00:00';threads='16';m='64g';vm='100g' # default 36hrs, 8 threads, 64gb, 100gb
    else
      wt='02:00:00';threads='16';m='32g';vm='64g'   # default 2hrs, 8 threads, 32gb, 64gb
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

conda init bash
conda activate abstract 
cd INSERT_SRA_ROOT_DIR

python3 run_models.py \
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