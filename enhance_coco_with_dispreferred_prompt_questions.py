import os
import sys
import copy
from io import BytesIO
import json
from PIL import Image
import random
import re
import requests
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from nltk import edit_distance
from sentence_transformers import SentenceTransformer
from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForQuestionAnswering,
    ViltProcessor,
    ViltForQuestionAnswering
)
from train_vqa_policy import ContextBuilder, answer_with_blip, answer_with_vilt, COCO_JSON_PATH, COCO_ENHANCED_JSON_PATH, BASE_DIR
from winoground_inference_with_LLM_fusion import ask_deepseek


def load_image_from_url(image_url: str) -> Image.Image:
    """Load image from URL with error handling"""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error loading image from {image_url}: {e}")
        return Image.new('RGB', (224, 224))  # Return blank image as fallback


def generate_neg_prompt(prompt):
    prompt_to_generate_description = f'''
        Given an input sentence describing a scene,
      your task is to first locate two swappable
      noun phrases in the sentence, and then swap
      them to make a new sentence. The new sentence
      must meet the following three requirements:
      1. The new sentence must be describing a
      different scene from the input sentence.
      2. The new sentence must be fluent and
      grammatically correct.
      3. The new sentence must make logical sense.
      To complete the task, you should:
      1. Answer the question of whether generating
      such a new sentence is possible using Yes
      or No.
      2. Output the swappable noun phrases.
      3. Swap the selected noun phrases to generate
      a new sentence.

      Here is an example input-output pair:

      Input: A woman cutting into a cake with a man
      standing behind her.
      Output: A man cutting into a cake with a
      woman standing behind him.

      If it is not possible to do so, locate the main subject
      of the sentence and change it with something that makes the sentence meaning different (like opposites, different breed/gender etc).

      Indicate your final response as:
      #Final Sentence#: Your final response here.

      The input is:{prompt}

                                      '''

    result = ask_deepseek(prompt_to_generate_description)
    return result.strip()


#def generate_checklist(text):
#    prompt = f"""
#    You are an AI assistant that generates a checklist of questions for evaluating images based on text descriptions. Given the following text description, generate a checklist of questions that focus on:
#    1. Objects: Identify the objects mentioned in the text.
#    2. Attributes: Describe the attributes (e.g., color, size, shape) of the objects.
#    3. Relations: Describe the relationships (e.g., proximity, position, interaction) between the objects.
#    Only generate true/false questions and the ground truth answers. Do not generate more questions than necessary.
#
#    Once you are done with the questions, give its negations as well such th.
#
#    Give your answer in the following format:
#    (The original Checklist)
#    1. Question 1 <True>
#    2. Question 2 <True>
#    ...
#    (The negative version of the original Checklist)
#    1. Question 1 <False>
#    2. Question 2 <False>
#
#    Here is an example output:
#    (The original Checklist)
#    1. Is there a man in the image? <True>
#    2. Is the man riding a skateboard? <True>
#    3. Is the skateboard in mid-air? <True>
#    4. Is there a pool in the image? <True>
#    5. Is the pool below the man? <True>
#    6. Is the man in a skateboard park? <True>
#    7. Is the man interacting with the skateboard? <True>
#    8. Is the man in a jumped position? <True>
#
#    (The negative version of the original Checklist)
#    1. Is there a woman in the image? <False>
#    2. Is the man not riding a skateboard? <False>
#    3. Is the skateboard not in mid-air? <False>
#    4. Is there a pool not in the image? <False>
#    5. Is the pool not below the man? <False>
#    6. Is the man not in a skateboard park? <False>
#    7. Is the man not interacting with the skateboard? <False>
#    8. Is the man not in a jumped position? <False>
#
#
#    Text: "{text}"
#
#    Checklist Questions:
#    """
#    checklist = ask_deepseek(prompt)
#    return checklist.strip().split("\n")


#def generate_checklist(dispreferred_prompt, original_prompt):
#    prompt = f"""
#    You are an AI assistant that generates a checklist of questions for evaluating images based on text descriptions. Given a text descriptions <good_text>, and an altered description <bad_text>, generate a checklist of questions that are true for <good_text> (but not <bad_text>) that focus on:
#    1. Objects: Identify the objects mentioned in <good_text>.
#    2. Attributes: Describe the attributes (e.g., color, size, shape) of the objects in <good_text>.
#    3. Relations: Describe the relationships (e.g., proximity, position, interaction) between the objects in <good_text>.
#    These should be true/false questions where the answer is true. Do not generate more questions than necessary.
#    Most importantly, do NOT generate any questions that are true for <bad_text>.
#    For example, if <good_text> is "A cat is eating a pizza" and <bad_text> is "A french fry is eaten by a cat", then you should generate "Is there a pizza in the image?" but NOT "Is there a cat in the image?" or "Is the cat eating?", as those are true for <bad_text>.
#
#    Once you are done with the questions, also give their negations such that the answer is false.
#
#    Give your answer in the following format:
#    (The original Checklist)
#    1. Question 1 <True>
#    2. Question 2 <True>
#    ...
#    (The negative version of the original Checklist)
#    1. Question 1 <False>
#    2. Question 2 <False>
#
#    Here is an example output:
#    (The original Checklist)
#    1. Is there a man in the image? <True>
#    2. Is the man riding a skateboard? <True>
#    3. Is the skateboard in mid-air? <True>
#    4. Is there a pool in the image? <True>
#    5. Is the pool below the man? <True>
#    6. Is the man in a skateboard park? <True>
#    7. Is the man interacting with the skateboard? <True>
#    8. Is the man in a jumped position? <True>
#
#    (The negative version of the original Checklist)
#    1. Is there a woman in the image? <False>
#    2. Is the man not riding a skateboard? <False>
#    3. Is the skateboard not in mid-air? <False>
#    4. Is there a pool not in the image? <False>
#    5. Is the pool not below the man? <False>
#    6. Is the man not in a skateboard park? <False>
#    7. Is the man not interacting with the skateboard? <False>
#    8. Is the man not in a jumped position? <False>
#
#
#    <good_text>: "{dispreferred_prompt}"
#    <bad_text>: "{original_prompt}"
#
#    Checklist Questions:
#    """
#    checklist = ask_deepseek(prompt)
#    return checklist.strip().split("\n")


def generate_checklist(dispreferred_prompt, original_prompt):
    prompt = f"""
    You are an AI assistant that generates a checklist of questions for evaluating images based on text descriptions. Given a text descriptions <good_text>, and an altered description <bad_text>, generate a checklist of questions that are true for <good_text> (but not <bad_text>) that focus on:
    1. Objects: Identify the objects mentioned in <good_text>.
    2. Attributes: Describe the attributes (e.g., color, size, shape) of the objects in <good_text>.
    3. Relations: Describe the relationships (e.g., proximity, position, interaction) between the objects in <good_text>.
    These should be true/false questions where the answer is true. Do not generate more questions than necessary.
    Most importantly, do NOT generate any questions that are true for <bad_text>.
    For example, if <good_text> is "A cat is eating a pizza" and <bad_text> is "A french fry is eaten by a cat", then you should generate "Is there a pizza in the image?" but NOT "Is there a cat in the image?" or "Is the cat eating?", as those are true for <bad_text>.

    Give your answer in the following format:
    1. Question 1 <True>
    2. Question 2 <True>
    ...

    Here is an example output:
    1. Is there a man in the image? <True>
    2. Is the man riding a skateboard? <True>
    3. Is the skateboard in mid-air? <True>
    4. Is there a pool in the image? <True>
    5. Is the pool below the man? <True>
    6. Is the man in a skateboard park? <True>
    7. Is the man interacting with the skateboard? <True>
    8. Is the man in a jumped position? <True>

    <good_text>: "{dispreferred_prompt}"
    <bad_text>: "{original_prompt}"

    Checklist Questions:
    """
    checklist = ask_deepseek(prompt)
    return checklist.strip().split("\n")


def sanitize_question(question):
    q = re.sub(r'[^\w\s]', '', question.lower())
    q = ' '.join(q.split())
    return q


def save_new_data(new_data):
    with open(COCO_ENHANCED_JSON_PATH, 'w') as f:
        json.dump(new_data, f)


def enhance_coco_with_dispreferred_prompt_questions():
    with open(COCO_JSON_PATH, 'r') as f:
        data = json.load(f)

    #load tools 'n stuffs
    device='cuda'
    context_builder = ContextBuilder(device=device)
    processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model_blip = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
    processor_vilt = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model_vilt = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)
    vqa_processors = [processor_blip, processor_vilt]
    vqa_models = [model_blip, model_vilt]

    new_data = []
    start_index = 0
    if os.path.exists(COCO_ENHANCED_JSON_PATH):
        with open(COCO_ENHANCED_JSON_PATH, 'r') as f:
            new_data = json.load(f)

        print('oh goody! we found %d examples already processed!'%(len(new_data)))
        start_index = len(new_data)

    for t, datum in tqdm(enumerate(data[start_index:])):
        new_datum = copy.deepcopy(datum)

        #load stuff
        image = load_image_from_url(datum['coco_url'])
        prompt = datum['prompt']
        questions = datum['questions']

        #make dispreferred prompt
        print('making dispreferred prompt...')
        dispreferred_prompt = generate_neg_prompt(prompt)
        match = re.search(r"#Final Sentence#:\s*(.+)", dispreferred_prompt)
        dispreferred_prompt = match.group(1) if match else None
        new_datum['dispreferred_prompt'] = dispreferred_prompt
        new_datum['dispreferred_prompt_questions'] = []
        new_datum['dispreferred_prompt_questions_duplicate'] = []
        if dispreferred_prompt is None:
            print('!')
            new_data.append(new_datum)
            continue

        #generate questions (remember to flip their answers)
        #filter redundant questions in this stage
        print('generating questions...')
        existing_questions = [sanitize_question(q['question']) for q in datum['questions']]
        dispreferred_checklist = generate_checklist(dispreferred_prompt, prompt)
        dispreferred_questions = []
        for item in dispreferred_checklist:
            if '<True>' in item or '<False>' in item:
                qa_doc = {}
                splitted = item.split('.')
                splitted = splitted[1].split('<')
                question = splitted[0]
                sanitized_question = sanitize_question(question)
                if any([sanitized_question == q for q in existing_questions]):
                    new_datum['dispreferred_prompt_questions_duplicate'].append(question)
                    continue

                answer = splitted[1][:-1]
                qa_doc['question'] = question
                qa_doc['answer'] = 1 if answer == 'True' else 0
                qa_doc['answer'] = 1 - qa_doc['answer'] #flip answer because it came from dispreferred prompt
                dispreferred_questions.append(qa_doc)

        #generate context and query tools
        print('context and queries...')
        for question_obj in dispreferred_questions:
            question = question_obj['question']
            ground_truth = question_obj['answer']
            context = context_builder.build(image, question)
            question_obj['context'] = context.tolist()
            vilt_pred = answer_with_vilt(image, question + ' Only answer by saying yes or no.', processor_vilt, model_vilt, device)
            vilt_pred = 1 if str(vilt_pred).lower() == 'yes' else 0 if str(vilt_pred).lower() == 'no' else -1
            vilt_reward = 1 if vilt_pred == ground_truth else -1
            blip_pred = answer_with_blip(image, question + ' Only answer by saying yes or no.', processor_blip, model_blip, device)
            blip_pred = 1 if str(blip_pred).lower() == 'yes' else 0 if str(blip_pred).lower() == 'no' else -1
            blip_reward = 1 if blip_pred == ground_truth else -1
            question_obj['vilt_pred'] = vilt_pred
            question_obj['vilt_reward'] = vilt_reward
            question_obj['blip_pred'] = blip_pred
            question_obj['blip_reward'] = blip_reward

        print('done!')

        new_datum['dispreferred_prompt_questions'] = dispreferred_questions
        new_data.append(new_datum)
        if t % 5 == 0 and t > 1:
            save_new_data(new_data)

    save_new_data(new_data)


def usage():
    print('Usage: python enhance_coco_with_dispreferred_prompt_questions.py')


if __name__ == '__main__':
    enhance_coco_with_dispreferred_prompt_questions()
