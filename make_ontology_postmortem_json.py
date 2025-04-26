import os
import sys
from collections import deque
import json
from tqdm import tqdm
from influence_on_human_ratings import load_human_ratings_data, adjust_impath, GENERATORS, BASE_DIR
from influence_on_tifa_residuals import load_tifa_results_dict
from ontology_questions import get_log_filename
from ontology_eval import SKIP_FILENAMES


#d[k]['prompt']
#d[k]['images'][generator]['bbox_vis_filename']
#d[k]['images'][generator]['ontology_log_filename']
#d[k]['images'][generator]['tifa_info']
#d[k]['images'][generator]['human_rating']
def make_ontology_postmortem_json():
    ontology_postmortem_dict = {}
    d_human = load_human_ratings_data()
    d_tifa = load_tifa_results_dict()
    for k in tqdm(sorted(d_human.keys())):
        prompt = d_human[k]['prompt']
        ontology_postmortem_dict[k] = {'prompt' : prompt, 'images' : {}}
        for generator in GENERATORS:
            image_filename = adjust_impath(d_human[k]['images'][generator]['filename'])
            if image_filename not in d_tifa:
                continue

            human_score = d_human[k]['images'][generator]['rating']
            log_filename = get_log_filename(prompt, image_filename)
            if not os.path.exists(log_filename):
                continue

            if os.path.basename(log_filename) in SKIP_FILENAMES:
                continue

            ontology_postmortem_dict[k]['images'][generator] = {}
            bbox_vis_filename = os.path.join(os.path.dirname(log_filename),'bbox_vis',os.path.splitext(os.path.basename(log_filename))[0]+'.png')
            ontology_postmortem_dict[k]['images'][generator]['human_rating'] = human_score
            ontology_postmortem_dict[k]['images'][generator]['tifa_info'] = d_tifa[image_filename]
            ontology_postmortem_dict[k]['images'][generator]['ontology_log_filename'] = os.path.join('genaibench_ontology_logs', os.path.basename(log_filename))
            ontology_postmortem_dict[k]['images'][generator]['bbox_vis_filename'] = os.path.join('genaibench_ontology_logs/bbox_vis', os.path.basename(bbox_vis_filename))

    ontology_postmortem_dict_filename = os.path.join(BASE_DIR, 'ontology_postmortem_dict.json')
    with open(ontology_postmortem_dict_filename, 'w') as f:
        json.dump(ontology_postmortem_dict, f)


if __name__ == '__main__':
    make_ontology_postmortem_json()
