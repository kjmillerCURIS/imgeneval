import os
import sys
import pickle
from tqdm import tqdm
sys.path.append('tifa')
from tifascore import get_question_and_answers, filter_question_and_answers, UnifiedQAModel, tifa_score_single, VQAModel
import openai
from openai_utils import OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
from influence_on_human_ratings import load_human_ratings_data, adjust_impath, BASE_DIR
from statistics import mean
import numpy as np
import copy


STRIDE = 8
SAVE_FREQ = 10
SKIP_IF_PRESENT = True
BAD_KEYS = [1356, 1362, 1376] #[1356, 1368, 1376]


def is_weird(result):
    return (result['tifa_score'] != mean([result['question_details'][q]['scores'] for q in sorted(result['question_details'].keys())])) or (result['tifa_score'] != np.mean([result['question_details'][q]['scores'] for q in sorted(result['question_details'].keys())]))


#returns unifiedqa_model, vqa_model
def load_tifa_models():
    unifiedqa_model = UnifiedQAModel("allenai/unifiedqa-v2-t5-large-1363200")
    vqa_model = VQAModel("mplug-large")
    return unifiedqa_model, vqa_model


def run_tifa_on_genaibench(offset):
    offset = int(offset)
    assert(offset >= 0 and offset < STRIDE)

    d = load_human_ratings_data()
    unifiedqa_model, vqa_model = load_tifa_models()
    results_dict = {}
    results_dict_filename = os.path.join(BASE_DIR, 'tifa_results_on_genaibench_%d.pkl'%(offset))
    if os.path.exists(results_dict_filename):
        with open(results_dict_filename, 'rb') as f:
            results_dict = pickle.load(f)

    for i, k in tqdm(enumerate(sorted(d.keys())[offset::STRIDE])):
        assert(isinstance(k, int))
        if i > 0 and i % SAVE_FREQ == 0:
            with open(results_dict_filename, 'wb') as f:
                pickle.dump(results_dict, f)

        if k in BAD_KEYS:
            continue

        if SKIP_IF_PRESENT and all([adjust_impath(d[k]['images'][generator]['filename']) in results_dict for generator in sorted(d[k]['images'].keys())]):
            continue

        prompt = d[k]['prompt']
        print('MEOW')
        print(k)
        print('I SAID MEOW')
        gpt3_questions = get_question_and_answers(prompt)
        filtered_questions = filter_question_and_answers(unifiedqa_model, gpt3_questions)
        for generator in sorted(d[k]['images'].keys()):
            impath = d[k]['images'][generator]['filename']
            impath = adjust_impath(impath)
            if SKIP_IF_PRESENT and (impath in results_dict):
                continue

            result = tifa_score_single(vqa_model, copy.deepcopy(filtered_questions), impath)
            if is_weird(result):
                print('WEIRD: %s'%(impath))
                print(result)
                print('OH SO WEIRD')
                assert(False)

            results_dict[impath] = result

    with open(results_dict_filename, 'wb') as f:
        pickle.dump(results_dict, f)


def usage():
    print('Usage: python run_tifa_on_genaibench.py <offset>')


if __name__ == '__main__':
    run_tifa_on_genaibench(*(sys.argv[1:]))
