#!/bin/bash -l

#$ -P ivc-ml
#$ -l h_rt=0:59:59
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -pe omp 2
#$ -j y
#$ -m ea

module load miniconda
conda activate imgeneval-meow
cd ~/data/imgeneval
python finetune_text_embedder.py ${LOSS_TYPE} ${LR}

