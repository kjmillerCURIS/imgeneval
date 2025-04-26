#!/bin/bash -l

#$ -P ivc-ml
#$ -pe omp 2
#$ -l gpus=1
#$ -l gpu_c=6.0
#$ -l h_rt=5:59:59
#$ -j y
#$ -m ea

module load miniconda
conda activate imgeneval-meow
cd ~/data/imgeneval
python genaibench_questions_and_tools.py ${OFFSET} ${MODE}

