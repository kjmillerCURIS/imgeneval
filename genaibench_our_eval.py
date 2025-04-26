import os
import sys
import json
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
from influence_on_human_ratings import load_human_ratings_data, adjust_impath, GENERATORS
from influence_on_tifa_residuals import load_tifa_results_dict
from genaibench_inference_with_LLM_fusion import OUTPUT_PREFIX
from genaibench_questions_and_tools import DATA_STRIDE
from train_vqa_policy import BASE_DIR


def load_genaibench_output_data():
    genaibench_output_data = {}
    for offset in range(DATA_STRIDE):
        path = OUTPUT_PREFIX + '_%d.json'%(offset)
        with open(path, 'r') as f:
            d = json.load(f)

        for my_key in sorted(d.keys()):
            assert(my_key not in genaibench_output_data)
            genaibench_output_data[my_key] = d[my_key]

    return genaibench_output_data


def genaibench_our_eval():
    d_human = load_human_ratings_data()
    d_tifa = load_tifa_results_dict()
    genaibench_output_data = load_genaibench_output_data()
    human_scores = []
    tifa_scores = []
    our_scores = []
    for k in tqdm(sorted(d_human.keys())):
        for generator in GENERATORS:
            prompt = d_human[k]['prompt']
            image_filename = adjust_impath(d_human[k]['images'][generator]['filename'])
            human_score = d_human[k]['images'][generator]['rating']
            if image_filename not in d_tifa:
                print('!')
                continue

            tifa_score = d_tifa[image_filename]['tifa_score']
            if str((k, generator)) not in genaibench_output_data:
                print('!!')
                continue

            our_score = genaibench_output_data[str((k, generator))]['score']
            human_scores.append(human_score)
            tifa_scores.append(tifa_score)
            our_scores.append(our_score)

    print('Found %d GenAI-Bench examples'%(len(human_scores)))
    print('TIFA: spearman_rho = %f'%(spearmanr(human_scores, tifa_scores).correlation))
    print('Ours: spearman_rho = %f'%(spearmanr(human_scores, our_scores).correlation))


if __name__ == '__main__':
    genaibench_our_eval()
