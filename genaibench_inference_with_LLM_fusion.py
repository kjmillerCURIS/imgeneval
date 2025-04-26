import os
import sys
import json
import random
import requests
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print(f'hey lets make lots of fucking tools to do slightly different versions of the same thing ane make them all fucking incompatible with each other cuz were so fucking smart, anyway, "{e}"')

from train_vqa_policy import QNetwork, INPUT_DIM, ACTION_DIM, BASE_DIR
from genaibench_questions_and_tools import OUTPUT_PREFIX as INTERMEDIATE_PREFIX
from winoground_inference_with_LLM_fusion import ask_deepseek


#old policy model (late 3/4/2025 and/or early 3/5/2025)
#HIDDEN_DIMS = [128, 64]
#POLICY_MODEL_PATH = os.path.join(BASE_DIR, 'trained_q_network_4_action_epoch_99.pth')
#OUTPUT_PREFIX = os.path.join(BASE_DIR, 'genaibench_outputs_cossim_abstention_4models')

#new policy model (midday 3/5/2025)
#HIDDEN_DIMS = [256, 128, 64]
#POLICY_MODEL_PATH = os.path.join(BASE_DIR, 'trained_q_network_4_action_final_epoch_494.pth')
#OUTPUT_PREFIX = os.path.join(BASE_DIR, 'genaibench_outputs_cossim_abstention_4models_betterpolicy')


#very old 2-model policy model (1pm 3/5/2025)
#HIDDEN_DIMS = [128, 64]
#ACTION_DIM = 2
#POLICY_MODEL_PATH = os.path.join(BASE_DIR, 'trained_q_network_vanilla_supervision_hidden_dims_128_64/trained_q_network_vanilla_supervision_hidden_dims_128_64_epoch_70.pth')
#OUTPUT_PREFIX = os.path.join(BASE_DIR, 'genaibench_outputs_cossims_abstention_2models')


#very old 2-model policy model but with ontology questions (evening 3/5/2025)
#HIDDEN_DIMS = [128, 64]
#ACTION_DIM = 2
#POLICY_MODEL_PATH = os.path.join(BASE_DIR, 'trained_q_network_vanilla_supervision_hidden_dims_128_64/trained_q_network_vanilla_supervision_hidden_dims_128_64_epoch_70.pth')
#OUTPUT_PREFIX = os.path.join(BASE_DIR, 'genaibench_outputs_cossims_abstention_2models_ontology_questions')


#very old 2-model policy model but extra descriptive relationship questions (evening 3/6/2025)
#HIDDEN_DIMS = [128, 64]
#ACTION_DIM = 2
#POLICY_MODEL_PATH = os.path.join(BASE_DIR, 'trained_q_network_vanilla_supervision_hidden_dims_128_64/trained_q_network_vanilla_supervision_hidden_dims_128_64_epoch_70.pth')
#OUTPUT_PREFIX = os.path.join(BASE_DIR, 'genaibench_outputs_cossims_abstention_2models_extra_descriptive_relationship_questions')


#very old 2-model policy model with original questions but use gt answers to generate description to use in place of prompt (late night 3/6/2025)
HIDDEN_DIMS = [128, 64]
ACTION_DIM = 2
POLICY_MODEL_PATH = os.path.join(BASE_DIR, 'trained_q_network_vanilla_supervision_hidden_dims_128_64/trained_q_network_vanilla_supervision_hidden_dims_128_64_epoch_70.pth')
GENERATE_DESCRIPTION_FROM_GT_ANSWERS = True
OUTPUT_PREFIX = os.path.join(BASE_DIR, 'genaibench_outputs_cossims_abstention_2models_generate_description_from_gt_answers')


def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)

    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def setup_tools():
    device='cuda'
    policy_model = QNetwork(INPUT_DIM, HIDDEN_DIMS, ACTION_DIM).to(device)
    try:
        checkpoint = torch.load(POLICY_MODEL_PATH, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {weight_path} does not exist.")

    # Check if the checkpoint contains the model's state_dict
    if "model_state_dict" not in checkpoint:
        raise KeyError("The checkpoint does not contain 'model_state_dict'.")

    # Load the state_dict into the model
    policy_model.load_state_dict(checkpoint["model_state_dict"])

    # Set the model to evaluation mode
    policy_model.eval()

    # text encoder
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))

    return {'policy_model' : policy_model, 'text_encoder' : text_encoder}


def genaibench_inference_with_LLM_fusion_one(intermediate_datum, tools):
    device='cuda'
    output_datum = {'prompt' : intermediate_datum['prompt']}
    checklist_trues = []
    checklist_false = []
    no_decision = 0
    for question in intermediate_datum['questions']:
        question_text = question['question']
        context = question['context']
        q_values = tools['policy_model'](torch.FloatTensor(context).to(device))

        #abstention part
        if torch.all(q_values < 0):
            no_decision += 1
            continue

        action = torch.argmax(q_values).item()
        predicted = question[['blip_pred', 'vilt_pred', 'ofabase_pred', 'ofalarge_pred'][action]]
        if predicted == 1:
            checklist_trues.append(question_text)
        else:
            if predicted not in [0, -1]:
                print('INTERESTING: predicted=' + str(predicted))

            checklist_false.append(question_text)

    prompt_for_description_generation = (
        "Generate a short, factual sentence describing only the objects and attributes that are marked True "
        "from the pipeline’s detection. Do not infer or add any extra details.\n"
        "Detection Results:\n"
        + "\n".join(f"{item} (True)" for item in checklist_trues)
    )

    prompt_for_description_generation = (
        prompt_for_description_generation
        + "\n".join(f"{item} (False)" for item in checklist_false)
    )

    LLM_generated_description = ask_deepseek(prompt_for_description_generation)
    print(checklist_trues==[], checklist_false==[])
    print(LLM_generated_description)
    LLM_generated_description_embedding = tools['text_encoder'].encode(LLM_generated_description)
    text_for_prompt_embedding = intermediate_datum['prompt']
    if GENERATE_DESCRIPTION_FROM_GT_ANSWERS:
        gt_trues = [q['question'] for q in intermediate_datum['questions'] if q['answer'] == 1]
        gt_false = [q['question'] for q in intermediate_datum['questions'] if q['answer'] != 1]
        gt_prompt = (
            "Generate a short, factual sentence describing only the objects and attributes that are marked True "
            "from the pipeline’s detection. Do not infer or add any extra details.\n"
            "Detection Results:\n"
            + "\n".join(f"{item} (True)" for item in gt_trues)
        )
        gt_prompt = (
            gt_prompt
            + "\n".join(f"{item} (False)" for item in gt_false)
        )
        text_for_prompt_embedding = ask_deepseek(gt_prompt)
        output_datum['text_for_prompt_embedding'] = text_for_prompt_embedding

    prompt_embedding = tools['text_encoder'].encode(text_for_prompt_embedding)
    score = cosine_similarity(LLM_generated_description_embedding, prompt_embedding)
    output_datum['checklist_trues'] = checklist_trues
    output_datum['checklist_false'] = checklist_false
    output_datum['no_decision'] = no_decision
    output_datum['LLM_generated_description'] = LLM_generated_description
    output_datum['score'] = float(score)
    return output_datum


def save_output_data(genaibench_output_data, genaibench_output_filename):
    print('mrow!')
    with open(genaibench_output_filename, 'w') as f:
        json.dump(genaibench_output_data, f)


def genaibench_inference_with_LLM_fusion(offset):
    offset = int(offset)

    #setup policy model
    tools = setup_tools()

    #load data
    genaibench_intermediate_filename = INTERMEDIATE_PREFIX + '_%d.json'%(offset)
    with open(genaibench_intermediate_filename, 'r') as f:
        genaibench_intermediate_data = json.load(f)

    genaibench_output_filename = OUTPUT_PREFIX + '_%d.json'%(offset)
    genaibench_output_data = {}
    if os.path.exists(genaibench_output_filename):
        with open(genaibench_output_filename, 'r') as f:
            genaibench_output_data = json.load(f)

        print('oh goody, we already have %d (prompt, image) pairs!'%(len(genaibench_output_data)))


    for t, my_key in tqdm(enumerate(sorted(genaibench_intermediate_data.keys()))):
        print('t = %d'%(t))
        if my_key in genaibench_output_data:
            print('already have %s, skip'%(my_key))
            continue

        print('offset = %d'%(offset))
        intermediate_datum = genaibench_intermediate_data[my_key]
        output_datum = genaibench_inference_with_LLM_fusion_one(intermediate_datum, tools)
        genaibench_output_data[my_key] = output_datum
        if t % 5 == 0 and t > 0:
            save_output_data(genaibench_output_data, genaibench_output_filename)

    save_output_data(genaibench_output_data, genaibench_output_filename)


def usage():
    print('Usage: python genaibench_inference_with_LLM_fusion.py <offset>')


if __name__ == '__main__':
    genaibench_inference_with_LLM_fusion(*(sys.argv[1:]))
