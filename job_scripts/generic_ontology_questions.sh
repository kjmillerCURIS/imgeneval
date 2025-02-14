#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 3
#$ -l gpus=1
#$ -l gpu_c=7.5
#$ -l h_rt=2:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate imgeneval
cd ~/data/imgeneval
python ontology_questions.py ${GENERATOR} ${OFFSET}

