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
from openai import OpenAI
from openai_utils import OPENAI_API_KEY
from huggingface_utils import HUGGINGFACE_API_KEY
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print(f'hey lets make lots of fucking tools to do slightly different versions of the same thing ane make them all fucking incompatible with each other cuz were so fucking smart, anyway, "{e}"')

from train_vqa_policy import QNetwork, INPUT_DIM, ACTION_DIM
from huggingface_hub import login


BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/imgeneval-data')
WINOGROUND_DATA_FILENAME = os.path.join(BASE_DIR, 'winoground_data.json')
EPOCH_LIST = [70]
ACTION_DIM = 2
USE_NEW_DESCRIPTION_GENERATION = True


client = OpenAI( api_key=HUGGINGFACE_API_KEY,
                 base_url="https://api.deepinfra.com/v1/openai",)


def load_winoground_data(json_path = WINOGROUND_DATA_FILENAME):
    with open(json_path, 'r') as f:
        data = json.load(f)

    return data


def ask_deepseek(prompt):
    response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=4096,
                stream=False,
            )
    return response.choices[0].message.content


def make_prompt_for_description_generation_old(checklist_trues, checklist_false):
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
    return prompt_for_description_generation


def make_prompt_for_description_generation_new(checklist_trues, checklist_false):
    new_prompt = ('''
          You are an AI assistant that receives a set of yes/no question–answer pairs about an image, along with ground-truth or model-predicted answers. Your tasks are:
          Perform a consistency check on the question–answer pairs. If answers are logically contradictory, you must resolve or discard the conflicting True item(s).
          Example Rule: If “Are there [X] in the image?” is answered “False,” then any claim like “Are the [X] doing Y?” cannot be “True.”
          Example Rule: If “Is [ACTION] happening?” is “False,” then any sub-question (e.g., “Who is doing [ACTION]?”) must also be “False.”
          Output a final “reconciled” set of question–answer pairs where contradictory answers have been removed or corrected.
          Generate a short factual summary (one or two sentences) that reflects only the remaining True statements after reconciliation.
          Do not include details marked false or discarded.
          Keep it concise and do not restate the entire prompt or caption verbatim.
          Emphasize important relations (e.g., who is doing what to whom, if relevant).
          Example
          Input Q–A pairs:
          “Are there leaves in the image?” → False
          “Are leaves being shed?” → True
          “Is there a cat in the image?” → False
          “Are the leaves falling upwards?” → False
          Step 1 (Consistency): Realize #1 and #2 contradict each other. You cannot have leaves being shed if there are no leaves.
          Choose to keep #1 as likely correct (False = no leaves). Force #2 to False or discard it.
          Step 2 (Reconciled Answers):“Are there leaves in the image?” → False
          “Are leaves being shed?” → False (changed from True)
          “Is there a cat in the image?” → False
          “Are the leaves falling upwards?” → False
          Step 3 (Summary):Since all final answers are False, the summary might be “No leaves or cats are present,” or simply “No relevant objects or actions confirmed.”
          You will apply similar logic for any set of Q–A pairs, then produce the final reconciled set plus a concise summary.
          Give your answer in the following format: #Answer: [The summary]
            '''+ "\n".join(f"{item} (True)" for item in checklist_trues)
             + "\n".join(f"{item} (False)" for item in checklist_false)
            + "\n")

    return new_prompt


def make_prompt_for_description_generation(checklist_trues, checklist_false):
    if USE_NEW_DESCRIPTION_GENERATION:
        return make_prompt_for_description_generation_new(checklist_trues, checklist_false)
    else:
        return make_prompt_for_description_generation_old(checklist_trues, checklist_false)


def load_model(weight_path, hidden_dims, device='cuda'):
    # Initialize the model
    model = QNetwork(INPUT_DIM, hidden_dims, ACTION_DIM).to(device)

    # Load the checkpoint
    try:
        checkpoint = torch.load(weight_path, map_location=device)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {weight_path} does not exist.")

    # Check if the checkpoint contains the model's state_dict
    if "model_state_dict" not in checkpoint:
        raise KeyError("The checkpoint does not contain 'model_state_dict'.")

    # Load the state_dict into the model
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set the model to evaluation mode
    model.eval()

    return model


def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)

    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def winoground_inference_with_LLM_fusion_one_epoch(learning_type, hidden_dims, include_dispreferred_prompt_questions, do_reshuffle, repindex, epoch, text_encoder):
    device = 'cuda'
    hidden_dims_str = '_'.join([str(h) for h in hidden_dims])
    model_prefix = f"trained_q_network_{learning_type}_hidden_dims_{hidden_dims_str}"
    if include_dispreferred_prompt_questions:
        model_prefix += "_trainwithdispreferredquestions"

    model_suffix = f"_epoch_{epoch}.pth"
    model_dir = os.path.join(BASE_DIR, model_prefix)
    model_path = os.path.join(model_dir, model_prefix + model_suffix)
    results_prefix = f"cossimexploration_repindex{repindex}_reshuffle{do_reshuffle}_winoground_results_LLM_fusion_{learning_type}_hidden_dims_{hidden_dims_str}"
    if include_dispreferred_prompt_questions:
        results_prefix += "_trainwithdispreferredquestions"

    results_suffix = f"_epoch_{epoch}.pth"
    results_path = os.path.join(model_dir, results_prefix + results_suffix)
    if os.path.exists(results_path):
        print(f'skipping epoch {epoch}, result file already exists!')
        return

    model = load_model(model_path, hidden_dims, device=device)
    data = load_winoground_data()
    results = {'aggregate' : {'correct_answers' : 0, 'incorrect_answers' : 0}, 'raw' : []}
    for example in tqdm(data):
        caption0_image0, caption0_image1, caption1_image0, caption1_image1 = None, None, None, None
        caption0_embedding = text_encoder.encode(example[0][0]['prompt'])
        caption1_embedding = text_encoder.encode(example[1][0]['prompt'])
        caption0_caption1 = cosine_similarity(caption0_embedding, caption1_embedding)
        wino_id_dict, checklist_trues_dict, checklist_false_dict, LLM_generated_description_dict, no_decision_dict = {}, {}, {}, {}, {}
        for caption_idx,caption in enumerate(example):
            for image_idx,image in enumerate(caption):
                questions = image['questions']
                wino_id_dict[(caption_idx, image_idx)] = image['wino_id']
                checklist_trues = []
                checklist_false = []
                no_decision = 0
                for question in questions:
                    question_text = question['question']
                    context = question['context']
                    q_values = model(torch.FloatTensor(context).to(device))
                    if torch.all(q_values < 0): #astention part
                        no_decision += 1
                        continue

                    action = torch.argmax(q_values).item()
                    predicted = question[['blip_pred', 'vilt_pred'][action]]
                    if predicted == 1:
                        checklist_trues.append(question_text)
                    else:
                        if predicted not in [0, -1]:
                            print('INTERESTING: predicted=' + str(predicted))

                        checklist_false.append(question_text)

                if do_reshuffle:
                    random.shuffle(checklist_trues)
                    random.shuffle(checklist_false)

                checklist_trues_dict[(caption_idx, image_idx)] = checklist_trues
                checklist_false_dict[(caption_idx, image_idx)] = checklist_false
                no_decision_dict[(caption_idx, image_idx)] = no_decision
                prompt_for_description_generation = make_prompt_for_description_generation(checklist_trues, checklist_false)
                LLM_generated_description = ask_deepseek(prompt_for_description_generation)
                if USE_NEW_DESCRIPTION_GENERATION:
                    LLM_generated_description = LLM_generated_description.split('#Answer:')[-1].strip()

                print(checklist_trues==[], checklist_false==[])
                print(LLM_generated_description)
                LLM_generated_description_dict[(caption_idx, image_idx)] = LLM_generated_description
                LLM_generated_description_embedding = text_encoder.encode(LLM_generated_description)
                if caption_idx == 0:
                    if image_idx == 0:
                        caption0_image0 = cosine_similarity(LLM_generated_description_embedding, caption0_embedding)
                    else:
                        caption0_image1 = cosine_similarity(LLM_generated_description_embedding, caption0_embedding)
                else:
                    if image_idx == 0:
                        caption1_image0 = cosine_similarity(LLM_generated_description_embedding, caption1_embedding)
                    else:
                        caption1_image1 = cosine_similarity(LLM_generated_description_embedding, caption1_embedding)

        if caption0_image0 > caption0_image1 and caption1_image0 < caption1_image1:
            results['aggregate']['correct_answers'] += 1
        else:
            results['aggregate']['incorrect_answers'] += 1

        print([[caption0_image0, caption0_image1], [caption1_image0, caption1_image1]])
        print(caption0_caption1)
        print('Correct Answers: ',results['aggregate']['correct_answers'])
        print('Incorrect Answers: ',results['aggregate']['incorrect_answers'])
        print('-'*50)

        result_one = {'caption0' : example[0][0]['prompt'],
                        'caption1' : example[1][0]['prompt'],
                        'caption0_image0_cossim' : caption0_image0,
                        'caption0_image1_cossim' : caption0_image1,
                        'caption1_image0_cossim' : caption1_image0,
                        'caption1_image1_cossim' : caption1_image1,
                        'caption0_caption1_cossim' : caption0_caption1,
                        'wino_id_dict' : wino_id_dict,
                        'checklist_trues_dict' : checklist_trues_dict,
                        'checklist_false_dict' : checklist_false_dict,
                        'no_decision_dict' : no_decision_dict,
                        'LLM_generated_description_dict' : LLM_generated_description_dict}

        results['raw'].append(result_one)

    print('FINAL Correct Answers: ',results['aggregate']['correct_answers'])
    print('FINAL Incorrect Answers: ',results['aggregate']['incorrect_answers'])
    print('-'*50)

    torch.save(results, results_path)


def winoground_inference_with_LLM_fusion(learning_type, hidden_dims, include_dispreferred_prompt_questions, do_reshuffle, repindex):
    hidden_dims = [int(h) for h in hidden_dims.split('_')]
    include_dispreferred_prompt_questions = int(include_dispreferred_prompt_questions)
    do_reshuffle = int(do_reshuffle)
    repindex = int(repindex)
    assert(learning_type in ['RL', 'vanilla_supervision'])

    device = 'cuda'
    text_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))

    for epoch in EPOCH_LIST:
        winoground_inference_with_LLM_fusion_one_epoch(learning_type, hidden_dims, include_dispreferred_prompt_questions, do_reshuffle, repindex, epoch, text_encoder)


def usage():
    print('Usage: python winoground_inference_with_LLM_fusion.py <learning_type> <hidden_dims> <include_dispreferred_prompt_questions> <do_reshuffle> <repindex>')


if __name__ == '__main__':
    winoground_inference_with_LLM_fusion(*(sys.argv[1:]))
