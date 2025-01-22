import os
import sys
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
from influence_on_human_ratings import load_human_ratings_data, BASE_DIR, IMAGE_DIR, GENERATORS, ALL_CONDITIONS, compute_influences, adjust_impath, beautifulprint
from run_tifa_on_genaibench import STRIDE
from make_examples_html import generate_html
from statistics import mean


BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/imgeneval-data')
IMAGE_DIR = os.path.join(BASE_DIR, 'GenAIBenchImages')
GENERATORS = ['DALLE_3', 'DeepFloyd_I_XL_v1', 'Midjourney_6', 'SDXL_2_1', 'SDXL_Base', 'SDXL_Turbo']
ALL_CONDITIONS = sorted(['Action Relation', 'Attribute', 'Comparison', 'Counting', 'Differentiation', 'Negation', 'Part Relation', 'Scene', 'Spatial Relation', 'Universal'])
CONDITIONS_TO_SHOW = ['Differentiation', 'Comparison', 'Negation', 'Spatial Relation', 'Scene']
NUM_EXAMPLES = 5
RANDOM_SEED = 0


#yeah, really...
def correct_tifa_scores(results_dict):
    new_dict = {}
    for impath in sorted(results_dict.keys()):
        result = results_dict[impath]
        scores = [result['question_details'][q]['scores'] for q in sorted(result['question_details'].keys())]
        if np.mean(scores) != result['tifa_score']:
            print('scores = ' + str(scores))
            print(str(np.mean(scores)) + ' vs ' + str(mean(scores)) + ' vs ' + str(result['tifa_score']))
            print(impath)
            print(result['question_details'][sorted(result['question_details'].keys())[0]]['caption'])

#        result['tifa_score'] = np.mean(scores)
#        new_dict[impath] = result

#    return new_dict


def load_tifa_results_dict():
    results_dict = {}
    for offset in range(STRIDE):
        with open(os.path.join(BASE_DIR, 'tifa_results_on_genaibench_%d.pkl'%(offset)), 'rb') as f:
            results_dict_one = pickle.load(f)

        for impath in sorted(results_dict_one.keys()):
            assert(impath not in results_dict)
            results_dict[impath] = results_dict_one[impath]

        correct_tifa_scores(results_dict)
    return results_dict


#return X, Y where X is binary with columns same order as ALL_CONDITIONS and Y has columns same order as GENERATORS
#d should come from load_human_ratings_data()
def preprocess_data(d):
    X, Y = [], []
    for k in tqdm(sorted(d.keys())):
        assert(all([c in ALL_CONDITIONS for c in d[k]['conditions']]))
        X_row = [int(c in d[k]['conditions']) for c in ALL_CONDITIONS]
        Y_row = [d[k]['images'][generator]['rating'] for generator in GENERATORS]
        X.append(X_row)
        Y.append(Y_row)

    return np.array(X), np.array(Y)


#return X, y_human, y_tifa
#X is binary, y_human and y_tifa are vectors (all generators clubbed together)
def preprocess(d, tifa_results_dict):
    X, y_human, y_tifa = [], [], []
    for k in tqdm(sorted(d.keys())):
        X_row = [int(c in d[k]['conditions']) for c in ALL_CONDITIONS]
        for generator in GENERATORS:
            impath = adjust_impath(d[k]['images'][generator]['filename'])
            if impath not in tifa_results_dict:
                print('!')
                continue

            X.append(X_row)
            y_human.append(d[k]['images'][generator]['rating'])
            y_tifa.append(tifa_results_dict[impath]['tifa_score'])

    return np.array(X), np.array(y_human), np.array(y_tifa)


def human_tifa_linreg(y_human, y_tifa):
    y_human_pad = np.hstack([y_human[:,np.newaxis], np.ones((len(y_human), 1))])
    w = np.linalg.pinv(y_human_pad.T @ y_human_pad) @ y_human_pad.T @ y_tifa[:,np.newaxis]
    [slope, intercept] = np.squeeze(w)
    R2 = np.corrcoef(y_human, y_tifa)[0,1] ** 2
    print('R2 = ' + str(R2))
    y_human_scaled = slope * y_human + intercept
    residuals = y_tifa - y_human_scaled
    plt.clf()
    plt.title('Human Rating vs TIFA (R2=%.3f)'%(R2))
    plt.xlabel('Human Rating')
    plt.ylabel('TIFA')
    plt.scatter(y_human, y_tifa)
    plt.plot([1, 5], [slope * 1 + intercept, slope * 5 + intercept])
    plt.savefig(os.path.join(BASE_DIR, 'human_vs_tifa.png'))
    plt.clf()
    return residuals, slope, intercept


def stitch_examples_one_condition(d, tifa_results_dict, c, slope, intercept):
    my_keys = [k for k in sorted(d.keys()) if c in d[k]['conditions']]
    my_keys = random.sample(my_keys, NUM_EXAMPLES)
    image_paths, below, captions = [], [], []
    output_file = os.path.join(BASE_DIR, 'GenAIBench_TIFA_examples', c.replace(' ', '_') + '.html')
    for k in my_keys:
        image_paths.append([os.path.join('../GenAIBenchImages', adjust_impath(d[k]['images'][generator]['filename'], full_path=False)) for generator in GENERATORS[::3]])
        below_one = []
        for generator in GENERATORS[::3]:
            impath = adjust_impath(d[k]['images'][generator]['filename'])
            result = tifa_results_dict[impath]
            s = '<br>'.join([q + ' ' + str(result['question_details'][q]['choices']) + ' (' + result['question_details'][q]['answer'] + ') ==> ' + result['question_details'][q]['multiple_choice_vqa'] for q in sorted(result['question_details'].keys())])
            s = s + '<br>tifa_score = %.3f'%(result['tifa_score'])
            s = s + '<br>human_score_scaled = %.3f'%(d[k]['images'][generator]['rating'] * slope + intercept)
            below_one.append(s)

        below.append(below_one)
        captions.append(d[k]['prompt'])

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    generate_html(image_paths, below, captions, output_file)


def stitch_examples(d, tifa_results_dict, slope, intercept):
    random.seed(RANDOM_SEED)
    for c in tqdm(CONDITIONS_TO_SHOW):
        stitch_examples_one_condition(d, tifa_results_dict, c, slope, intercept)


def influence_on_tifa_residuals():
    d = load_human_ratings_data()
    tifa_results_dict = load_tifa_results_dict()
    X, y_human, y_tifa = preprocess(d, tifa_results_dict)
    residuals, slope, intercept = human_tifa_linreg(y_human, y_tifa)
    influences = compute_influences(X, residuals)
    print('influences on TIFA - human_scaled:')
    beautifulprint(influences)
    influences_abs = compute_influences(X, np.fabs(residuals))
    print('influences on |TIFA - human_scaled|:')
    beautifulprint(influences_abs)
    stitch_examples(d, tifa_results_dict, slope, intercept)


if __name__ == '__main__':
    influence_on_tifa_residuals()
