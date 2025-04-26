#!/bin/bash -l

#$ -P ivc-ml
#$ -l h_rt=5:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate imgeneval-meow
cd ~/data/imgeneval
python generate_data_for_finetuning_text_embedder.py ${OFFSET}

