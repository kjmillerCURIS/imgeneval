#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pycocotools')
get_ipython().system('wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip')
get_ipython().system('unzip annotations_trainval2017.zip')


# In[ ]:


from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from openai import OpenAI


# In[ ]:


coco=COCO('/content/annotations/instances_train2017.json')
coco_caps=COCO('/content/annotations/captions_train2017.json')


# In[ ]:


cats = coco.loadCats(coco.getCatIds())
images = []
for cat in cats:
  imgIds = coco.getImgIds(catIds=cat['id'])
  rand_array = np.random.randint(0, len(imgIds), (1, 3))
  rand_indices = rand_array.flatten()
  random_img_ids = [imgIds[i] for i in rand_indices]
  images += coco.loadImgs(random_img_ids)


# In[ ]:


client = OpenAI( api_key="eQGEavzoH1vRZSgRJoLQAb2ZfzZpAYQk",
                 base_url="https://api.deepinfra.com/v1/openai",)

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

def generate_prompt(attr):
  prompt_to_generate_description = '''
        You are an expert prompt engineer for text-to-image models. Your task is to create a straightforward, factual description based on the following image components. Follow these guidelines:

        1. Combine phrases into a single concise paragraph
        2. Mention only explicitly specified visual elements - do not add style/mood/environment details unless provided
        3. Maintain neutral, objective language without artistic embellishment
        4. Remove all redundancy while preserving key elements
        5. Prioritize clarity over descriptive flair

        Include ONLY these elements from the provided phrases:
        '''
  for phrase in attr:
    prompt_to_generate_description += f"{phrase}\n"

  result = ask_deepseek(prompt_to_generate_description)
  return result.strip()

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


# In[ ]:


import json
import random
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from io import BytesIO
from collections import deque
from sentence_transformers import SentenceTransformer
from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForQuestionAnswering,
    ViltProcessor,
    ViltForQuestionAnswering
)
import re

text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Device Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Helper Function for Image Loading
def load_image_from_url(image_url: str) -> Image.Image:
    """Load image from URL with error handling"""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error loading image from {image_url}: {e}")
        return Image.new('RGB', (224, 224))  # Return blank image as fallback

# BLIP Implementation
processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model_blip = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)


def answer_with_blip(image: Image.Image, question: str):
    """Answer a VQA question using BLIP and return confidence with geometric mean normalization"""
    try:
        inputs = processor_blip(image, question, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model_blip.generate(**inputs, output_scores=True, return_dict_in_generate=True)

        # Decode answer
        answer = processor_blip.decode(output.sequences[0], skip_special_tokens=True)

        # Extract logits for generated tokens (excluding <BOS>)
        token_logits = output.scores  # List of logits for each generated token
        token_probs = [torch.softmax(logit, dim=-1).max().item() for logit in token_logits]

        # Compute geometric mean of token probabilities
        if token_probs:
            confidence = np.exp(np.mean(np.log(token_probs)))  # Geometric mean
        else:
            confidence = 0.0  # In case of empty output

        return answer, confidence
    except Exception as e:
        print(f"BLIP error: {e}")
        return "error", 0.0



processor_vilt = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model_vilt = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)

def answer_with_vilt(image: Image.Image, question: str):
    """Answer a VQA question using VILT and return the answer with confidence score"""
    try:
        encoding = processor_vilt(image, question, return_tensors="pt").to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model_vilt(**encoding)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        pred_idx = outputs.logits.argmax(-1).item()
        confidence = probs[0, pred_idx].item()  # Extract confidence score

        return model_vilt.config.id2label[pred_idx], confidence

    except Exception as e:
        print(f"VILT error: {e}")
        return "error", 0.0

class ContextBuilder:
    def __init__(self):
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def build(self, image: Image.Image, question: str) -> np.ndarray:
        """Build context vector from image and question"""
        try:
            # Text embedding
            text_embed = self.text_encoder.encode([question])[0]

            # Image embedding
            inputs = self.clip_processor(
                images=image,
                return_tensors="pt"
            )
            with torch.no_grad():
                image_embed = self.clip_model.get_image_features(**inputs)
                image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

            return np.concatenate([text_embed, image_embed.cpu().numpy().flatten()])
        except Exception as e:
            print(f"Context building error: {e}")
            return np.zeros(384 + 512)  # Return zero vector on error



def compute_cosine_similarity(text1: str, text2: str) -> float:
    """
    Compute the cosine similarity between two text inputs using a SentenceTransformer model.

    Args:
        text1 (str): First text input.
        text2 (str): Second text input.

    Returns:
        float: Cosine similarity score between -1 and 1.
    """
    try:
        # Encode the texts into vector embeddings
        embedding1 = text_encoder.encode(text1, convert_to_tensor=True)
        embedding2 = text_encoder.encode(text2, convert_to_tensor=True)

        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(embedding1, embedding2, dim=0)

        return cosine_sim.item()

    except Exception as e:
        print(f"Error computing similarity: {e}")
        return 0.0


data = []
for image_idx, image in enumerate(images):
  annIds = coco_caps.getAnnIds(imgIds=image['id']);
  anns = coco_caps.loadAnns(annIds)

  prompt = generate_prompt(anns[:1])

  checklist = generate_checklist(prompt)
  image_url = image['coco_url']
  image_id = image['id']

  image = load_image_from_url(image['coco_url'])

  inputs = clip_processor(
      images=image,
      return_tensors="pt"
  )
  with torch.no_grad():
    image_embed = clip_model.get_image_features(**inputs)
    image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)

  questions = []
  for item in checklist:
    if "<True>" in item or '<False>' in item:
      qa_doc = {}
      splitted = item.split('.')
      splitted = splitted[1].split('<')
      question = splitted[0]
      answer = splitted[1][:-1]
      qa_doc['question'] = question
      qa_doc['answer'] = 1 if answer == 'True' else 0
      questions.append(qa_doc)

  for question_obj in questions:
        question = question_obj['question']
        ground_truth = question_obj['answer']

        text_embed = text_encoder.encode([question])[0]

        context = np.concatenate([text_embed, image_embed.cpu().numpy().flatten()])
        question_obj['context'] = context.tolist()

        vilt_pred,vilt_prob = answer_with_vilt(image, question + ' Only answer by saying yes or no.')
        vilt_pred = 1 if str(vilt_pred).lower() == 'yes' else 0 if str(vilt_pred).lower() == 'no' else -1
        vilt_reward = 1 if vilt_pred == ground_truth else -1

        blip_pred,blip_prob = answer_with_blip(image, question + ' Only answer by saying yes or no.')
        blip_pred = 1 if str(blip_pred).lower() == 'yes' else 0 if str(blip_pred).lower() == 'no' else -1
        blip_reward = 1 if blip_pred == ground_truth else -1

        question_obj['vilt_reward'] = vilt_reward
        question_obj['vilt_pred'] = vilt_pred
        question_obj['vilt_prob'] = vilt_prob

        question_obj['blip_pred'] = blip_pred
        question_obj['blip_reward'] = blip_reward
        question_obj['blip_prob'] = blip_prob

        question_obj['cos_sim_with_prompt'] = compute_cosine_similarity(prompt,question)

  results = generate_neg_prompt(prompt)


  match = re.search(r"#Final Sentence#:\s*(.+)", results)
  negative_prompt = match.group(1) if match else None
  print(prompt)
  print(negative_prompt)

  negative_checklist = generate_checklist(negative_prompt)


  neg_questions = []
  for item in negative_checklist:
    if "<True>" in item or '<False>' in item:
      qa_doc = {}
      splitted = item.split('.')
      splitted = splitted[1].split('<')
      question = splitted[0]
      answer = splitted[1][:-1]
      qa_doc['question'] = question
      qa_doc['answer'] = 1 if answer == 'True' else 0
      neg_questions.append(qa_doc)

  for question_obj in neg_questions:
        question = question_obj['question']
        ground_truth = question_obj['answer']

        text_embed = text_encoder.encode([question])[0]

        context = np.concatenate([text_embed, image_embed.cpu().numpy().flatten()])
        question_obj['context'] = context.tolist()

        vilt_pred,vilt_prob = answer_with_vilt(image, question + ' Only answer by saying yes or no.')
        vilt_pred = 1 if str(vilt_pred).lower() == 'yes' else 0 if str(vilt_pred).lower() == 'no' else -1
        vilt_reward = 1 if vilt_pred == ground_truth else -1

        blip_pred,blip_prob = answer_with_blip(image, question + ' Only answer by saying yes or no.')
        blip_pred = 1 if str(blip_pred).lower() == 'yes' else 0 if str(blip_pred).lower() == 'no' else -1
        blip_reward = 1 if blip_pred == ground_truth else -1

        question_obj['vilt_reward'] = vilt_reward
        question_obj['vilt_pred'] = vilt_pred
        question_obj['vilt_prob'] = vilt_prob

        question_obj['blip_pred'] = blip_pred
        question_obj['blip_reward'] = blip_reward
        question_obj['blip_prob'] = blip_prob

        question_obj['cos_sim_with_prompt'] = compute_cosine_similarity(prompt,question)
        question_obj['cos_sim_with_neg_prompt'] = compute_cosine_similarity(negative_prompt,question)


  image_data = {
        "coco_url": image_url,
        "image_id": image_id,
        "captions": [ann['caption'] for ann in anns],
        "prompt": prompt,
        "checklist": checklist,
        "negative_prompt": negative_prompt,
        "negative_checklist": negative_checklist,
        "questions": questions,
        "negative_questions": neg_questions,
    }
  data.append(image_data)


# In[ ]:


import json
with open('coco_image_data_with_negatives.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Data saved to coco_image_data.json")


# In[ ]:


print(results)


# In[ ]:




