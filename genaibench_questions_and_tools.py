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
from torchvision import transforms
try:
    from transformers import OFATokenizer, OFAModel
except Exception as e:
    print(f'hey lets make lots of fucking tools to do slightly different versions of the same thing ane make them all fucking incompatible with each other cuz were so fucking smart, anyway, "{e}"')

from nltk import edit_distance
try:
    from transformers import (
        CLIPProcessor,
        CLIPModel,
        BlipProcessor,
        BlipForQuestionAnswering,
        ViltProcessor,
        ViltForQuestionAnswering
    )
except Exception as e:
    print(f'hey lets make lots of fucking tools to do slightly different versions of the same thing ane make them all fucking incompatible with each other cuz were so fucking smart, anyway, "{e}"')

from influence_on_human_ratings import load_human_ratings_data, adjust_impath
from train_vqa_policy import answer_with_blip, answer_with_vilt, answer_with_ofabase, answer_with_ofalarge, BASE_DIR, ContextBuilder
from winoground_inference_with_LLM_fusion import ask_deepseek
from generate_questions_from_semantic_graph import DATA_STRIDE as QUESTIONS_DATA_STRIDE
from generate_questions_from_semantic_graph import QUESTIONS_PREFIX


DATA_STRIDE = 12

#original questions (late 3/4/2025 and/or early 3/5/2025)
USE_ONTOLOGY_QUESTIONS = False
ADD_DESCRIPTION_RELATIONSHIP_QUESTIONS = False
OUTPUT_PREFIX = os.path.join(BASE_DIR, 'genaibench_intermediates_4models')

#ontology questions (evening 3/5/2025)
#USE_ONTOLOGY_QUESTIONS = True
#OUTPUT_PREFIX = os.path.join(BASE_DIR, 'genaibench_intermediates_4models_ontology_questions')

#descriptive relationship questions (early 3/6/2025)
#USE_ONTOLOGY_QUESTIONS = False
#ADD_DESCRIPTIVE_RELATIONSHIP_QUESTIONS = True
#OUTPUT_PREFIX = os.path.join(BASE_DIR, 'genaibench_intermediates_4models_extra_descriptive_relationship_questions')


def generate_checklist(text):
    prompt = f"""
    You are an AI assistant that generates a checklist of questions for evaluating images based on text descriptions. Given the following text description, generate a checklist of questions that focus on:
    1. Objects: Identify the objects mentioned in the text.
    2. Attributes: Describe the attributes (e.g., color, size, shape) of the objects.
    3. Relations: Describe the relationships (e.g., proximity, position, interaction) between the objects.
    Only generate true/false questions and the ground truth answers. Do not generate more questions than necessary.

    Once you are done with the questions, give its negations as well such th.

    Give your answer in the following format:
    (The original Checklist)
    1. Question 1 <True>
    2. Question 2 <True>
    ...
    (The negative version of the original Checklist)
    1. Question 1 <False>
    2. Question 2 <False>

    Here is an example output:
    (The original Checklist)
    1. Is there a man in the image? <True>
    2. Is the man riding a skateboard? <True>
    3. Is the skateboard in mid-air? <True>
    4. Is there a pool in the image? <True>
    5. Is the pool below the man? <True>
    6. Is the man in a skateboard park? <True>
    7. Is the man interacting with the skateboard? <True>
    8. Is the man in a jumped position? <True>

    (The negative version of the original Checklist)
    1. Is there a woman in the image? <False>
    2. Is the man not riding a skateboard? <False>
    3. Is the skateboard not in mid-air? <False>
    4. Is there a pool not in the image? <False>
    5. Is the pool not below the man? <False>
    6. Is the man not in a skateboard park? <False>
    7. Is the man not interacting with the skateboard? <False>
    8. Is the man not in a jumped position? <False>


    Text: "{text}"

    Checklist Questions:
    """
    checklist = ask_deepseek(prompt)
    return checklist.strip().split("\n")


def generate_extra_checklist(prompt, checklist):
    checklist_input_str = '\n'.join([c for c in checklist if '<True>' in c or '<False>' in c])
    prompt = f"""
    You are an AI assistant that generates a checklist of questions for evaluating images based on text descriptions. Given a text description, another AI assistant has generated true/false questions that inquired about the presence of obejects, their attributes, and the relationships between them. You must now make the relationship questions more descriptive in the following way:
        a. Active Role (Subject) Cues:
            * “Is [ENTITY1] (the subject) exhibiting active motion or gestures (e.g., leaning in, extended arm, forward head movement) toward [ENTITY2]?”
            * Optionally, include a negative check for [ENTITY2] to reinforce role differentiation: “Is [ENTITY2] (the object) not exhibiting active motion indicative of initiating the [ACTION]?”
        b. Spatial and Contact Cues:
            * “Are [ENTITY1] and [ENTITY2] in close physical proximity (touching or nearly touching)?”
            * “Is the relevant body part of [ENTITY1] (e.g., arm, branch) in contact with [ENTITY2]?”

    For example, given the relationship question "Is the car crashing into the tree?", you could generate some or all of the following descriptive questions:
        * "Is the tree (subject) moving toward the car with an active motion (e.g., falling or crashing)?"
        * "Is the car (object) not exhibiting active motion indicative of initiating the smash?"
        * "Are the tree and the car in close physical proximity (touching or nearly touching)?"
        * "Is a branch or trunk of the tree in contact with the car?"

    Please go through the checklist of questions provided, pick out the ones that are relationships, and generate descriptive versions of them. Generate no more than 5 of these descriptive questions TOTAL. Follow the same output format as the input checklist provided, including the answer <True> or <False> after each question.

    Text Description: "{prompt}"

    Input Checklist Questions:
    {checklist_input_str}

    Descriptive Relationship Checklist Questions:
    """
    checklist = ask_deepseek(prompt)
    return checklist.strip().split("\n")


def setup_tools(mode):
    device='cuda'
    if mode == 'not_OFA':
        context_builder = ContextBuilder(device=device)
        processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model_blip = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)
        processor_vilt = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        model_vilt = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)

    if mode == 'OFA':
        mean_ofa, std_ofa = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        resolution_ofabase = 384
        patch_transform_ofabase = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution_ofabase, resolution_ofabase), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_ofa, std=std_ofa)
        ])
        tokenizer_ofabase = OFATokenizer.from_pretrained('OFA-base')
        model_ofabase = OFAModel.from_pretrained('OFA-base', use_cache=False).to(device)
        resolution_ofalarge = 384
        patch_transform_ofalarge = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize((resolution_ofalarge, resolution_ofalarge), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_ofa, std=std_ofa)
        ])
        tokenizer_ofalarge = OFATokenizer.from_pretrained('OFA-large')
        model_ofalarge = OFAModel.from_pretrained('OFA-large', use_cache=False).to(device)

    if mode == 'not_OFA':
        tools = {'context_builder' : context_builder,
                'processor_blip' : processor_blip,
                'model_blip' : model_blip,
                'processor_vilt' : processor_vilt,
                'model_vilt' : model_vilt}

    if mode == 'OFA':
        tools = {'patch_transform_ofabase' : patch_transform_ofabase,
                'tokenizer_ofabase' : tokenizer_ofabase,
                'model_ofabase' : model_ofabase,
                'patch_transform_ofalarge' : patch_transform_ofalarge,
                'tokenizer_ofalarge' : tokenizer_ofalarge,
                'model_ofalarge' : model_ofalarge}

    return tools


#return datum
def genaibench_questions_and_tools_one(prompt, image_filename, tools, mode, output_datum=None, questions_datum=None):
    device='cuda'
    if mode == 'not_OFA':
        output_datum = {'prompt' : prompt, 'image_filename' : image_filename}

    image = Image.open(image_filename)

    #generate questions
    if mode == 'not_OFA':
        if USE_ONTOLOGY_QUESTIONS:
            questions = copy.deepcopy(questions_datum)
        else:
            print('generating questions...')
            checklist = generate_checklist(prompt)
            output_datum['checklist'] = checklist
            all_checklist = checklist
            if ADD_DESCRIPTIVE_RELATIONSHIP_QUESTIONS:
                extra_checklist = generate_extra_checklist(prompt, checklist)
                print(prompt)
                print(checklist)
                print(extra_checklist)
                output_datum['extra_checklist'] = extra_checklist
                all_checklist = checklist + extra_checklist

            questions = []
            for item in all_checklist:
                if '<True>' in item or '<False>' in item:
                    qa_doc = {}
                    splitted = item.split('.')
                    splitted = splitted[-1].split('<')
                    question = splitted[0]
                    answer = splitted[-1][:-1]
                    qa_doc['question'] = question
                    qa_doc['answer'] = 1 if answer == 'True' else 0
                    questions.append(qa_doc)

    if mode == 'OFA':
        questions = output_datum['questions']

    #context and tools
    print('context and tools...')
    for question_obj in questions:
        question = question_obj['question']
        ground_truth = question_obj['answer']
        if mode == 'not_OFA':
            context = tools['context_builder'].build(image, question)
            question_obj['context'] = context.tolist()
            vilt_pred = answer_with_vilt(image, question + ' Only answer by saying yes or no.', tools['processor_vilt'], tools['model_vilt'], device)
            vilt_pred = 1 if str(vilt_pred).lower() == 'yes' else 0 if str(vilt_pred).lower() == 'no' else -1
            vilt_reward = 1 if vilt_pred == ground_truth else -1
            blip_pred = answer_with_blip(image, question + ' Only answer by saying yes or no.', tools['processor_blip'], tools['model_blip'], device)
            blip_pred = 1 if str(blip_pred).lower() == 'yes' else 0 if str(blip_pred).lower() == 'no' else -1
            blip_reward = 1 if blip_pred == ground_truth else -1
            question_obj['vilt_pred'] = vilt_pred
            question_obj['vilt_reward'] = vilt_reward
            question_obj['blip_pred'] = blip_pred
            question_obj['blip_reward'] = blip_reward

        if mode == 'OFA':
            ofabase_pred = answer_with_ofabase(image, question + ' Only answer by saying yes or no.', tools['patch_transform_ofabase'], tools['tokenizer_ofabase'], tools['model_ofabase'], device)
            ofabase_pred = 1 if str(ofabase_pred).lower() == 'yes' else 0 if str(ofabase_pred).lower() == 'no' else -1
            ofabase_reward = 1 if ofabase_pred == ground_truth else -1
            ofalarge_pred = answer_with_ofalarge(image, question + ' Only answer by saying yes or no.', tools['patch_transform_ofalarge'], tools['tokenizer_ofalarge'], tools['model_ofalarge'], device)
            ofalarge_pred = 1 if str(ofalarge_pred).lower() == 'yes' else 0 if str(ofalarge_pred).lower() == 'no' else -1
            ofalarge_reward = 1 if ofalarge_pred == ground_truth else -1
            question_obj['ofabase_pred'] = ofabase_pred
            question_obj['ofabase_reward'] = ofabase_reward
            question_obj['ofalarge_pred'] = ofalarge_pred
            question_obj['ofalarge_reward'] = ofalarge_reward

    #package datum and return
    output_datum['questions'] = questions
    return output_datum


def save_output_data(genaibench_output_data, genaibench_output_filename):
    print('mrow!')
    with open(genaibench_output_filename, 'w') as f:
        json.dump(genaibench_output_data, f)


def load_questions_data():
    genaibench_questions_data = {}
    for offset in range(QUESTIONS_DATA_STRIDE):
        with open(QUESTIONS_PREFIX + '_%d.json'%(offset), 'r') as f:
            d = json.load(f)
            for my_key in sorted(d.keys()):
                assert(my_key not in genaibench_questions_data)
                genaibench_questions_data[my_key] = d[my_key]

    return genaibench_questions_data


def genaibench_questions_and_tools(offset, mode):
    offset = int(offset)
    assert(mode in ['not_OFA', 'OFA'])

    #setup tools
    tools = setup_tools(mode)

    #load data
    #d[k] = {'prompt' : prompt, 'conditions' : conditions, 'images' : {generator : {'filename' : filename, 'rating' : rating}}}
    genaibench_input_data = load_human_ratings_data()
    if USE_ONTOLOGY_QUESTIONS:
        genaibench_questions_data = load_questions_data()

    if mode == 'not_OFA':
        genaibench_output_filename = OUTPUT_PREFIX + '_withoutOFA_%d.json'%(offset)
    else:
        assert(mode == 'OFA')
        genaibench_inter_filename = OUTPUT_PREFIX + '_withoutOFA_%d.json'%(offset)
        genaibench_output_filename = OUTPUT_PREFIX + '_%d.json'%(offset)

    genaibench_output_data = {}
    if os.path.exists(genaibench_output_filename):
        with open(genaibench_output_filename, 'r') as f:
            genaibench_output_data = json.load(f)

        print('oh goody, we already have %d (prompt, image) pairs!'%(len(genaibench_output_data)))

    if mode == 'OFA':
        with open(genaibench_inter_filename, 'r') as f:
            genaibench_inter_data = json.load(f)

    for t, k in tqdm(enumerate(sorted(genaibench_input_data.keys())[offset::DATA_STRIDE])):
        print('t = %d'%(t))
        prompt = genaibench_input_data[k]['prompt'] #this comes directly from the GenAIBench dataset
        for generator in sorted(genaibench_input_data[k]['images'].keys()):
            if str((k, generator)) in genaibench_output_data:
                print('already have %s, skip'%(str((k, generator))))
                continue

            image_filename = adjust_impath(genaibench_input_data[k]['images'][generator]['filename'])
            if not os.path.exists(image_filename):
                print('missing image "%s", skip'%(image_filename))
                continue

            if USE_ONTOLOGY_QUESTIONS:
                if str((k, generator)) not in genaibench_questions_data:
                    print('missing ontology questions for %s, skip'%(str((k, generator))))
                    continue

            print('offset = %d'%(offset))
            if mode == 'OFA': #no need to pass in ontology questions in this case, they're already in inter_datum
                inter_datum = genaibench_inter_data[str((k, generator))]
                output_datum = genaibench_questions_and_tools_one(prompt, image_filename, tools, mode, output_datum=inter_datum)
            elif USE_ONTOLOGY_QUESTIONS:
                assert(mode == 'not_OFA')
                questions_datum = genaibench_questions_data[str((k, generator))]
                output_datum = genaibench_questions_and_tools_one(prompt, image_filename, tools, mode, questions_datum=questions_datum)
            else:
                assert(mode == 'not_OFA')
                output_datum = genaibench_questions_and_tools_one(prompt, image_filename, tools, mode)

            genaibench_output_data[str((k, generator))] = output_datum

            if t % 5 == 0 and t > 0:
                save_output_data(genaibench_output_data, genaibench_output_filename)

    save_output_data(genaibench_output_data, genaibench_output_filename)


def usage():
    print('Usage: python genaibench_questions_and_tools.py <offset> <mode>')


if __name__ == '__main__':
    genaibench_questions_and_tools(*(sys.argv[1:]))
