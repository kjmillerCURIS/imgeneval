import os
import sys
import numpy as np
import pickle
import random
from tqdm import tqdm
import openai
from openai_utils import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY
from backtick import BACKTICK
from benchmark_negclip_on_our_synthetic_dataset import OUR_OUTPUT_FILENAME
from explore_our_synthetic_data_negclip_outputs import is_in_range


RANDOM_SEED = 0
NUM_EXAMPLES = 30
TRIPLE_TICKS = 3 * BACKTICK
ORIGINAL_PROMPT = '''
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

      The input is:{{ANCHOR}}
'''


def query_LLM(query):
    messages = [{"role": "system", "content": query}]
    response = openai.chat.completions.create(
        model="gpt-4o",  # Use the GPT-4 model
        messages=messages
    )
    reply = response.choices[0].message.content
    return reply


def form_examples_part(examples):
    lines = []
    for example in examples:
        lines.append('ANCHOR: "%s"'%(example['caption']))
        lines.append('T-: "%s"'%(example['negative_example']))
        lines.append('')

    return '\n'.join(lines)


def form_query(hard_examples, easy_examples):
    hard_examples_part = form_examples_part(hard_examples)
    easy_examples_part = form_examples_part(easy_examples)
    query = f'''
    You are a prompt-engineering expert. You are tasked with creating a challenging image-text training set to finetune VLLMs to make them better at Winoground-like tasks.
    A fellow prompt engineer started this task by taking a manually curated image-caption dataset (COCO), taking each caption (also referred to as "ANCHOR") and using an off-the-shelf LLM to generate a semantically different hard-negative version of the caption "T-". Here is the prompt that was given to the off-the-shelf LLM:
    {TRIPLE_TICKS}
    {ORIGINAL_PROMPT}
    {TRIPLE_TICKS}

    Your colleague then tested these anchors and synthetic hard-negatives against the VLLM and found that these were among the most challenging ones (i.e. the VLLM gave T- a high similarity score with the image, relative to ANCHOR with the image):

    {hard_examples_part}

    By contrast, these examples were among the easiest for the VLLM, and not very challenging:

    {easy_examples_part}

    Explain the patterns that you see in these hard and easy examples - what makes the hard ones harder than the easy ones? Then, please refine your colleague\'s LLM prompt so that it generates a more challenging T-'s. Please use triple-backticks ("{TRIPLE_TICKS}") before and after the refined LLM prompt, and remember to include "{{ANCHOR}}" in the refined prompt where the original caption would go.
    '''
    return query


def select_examples(outputs, min_perc, max_perc):
    vals = [output['negative_cossim'] - output['positive_cossim'] for output in outputs]
    min_thresh = np.percentile(vals, min_perc)
    max_thresh = np.percentile(vals, max_perc)
    selected_outputs = [output for output in outputs if is_in_range(output['negative_cossim'] - output['positive_cossim'], min_thresh, max_thresh)]
    selected_outputs = random.sample(selected_outputs, NUM_EXAMPLES)
    return selected_outputs


def LLM_tweak_prompt(min_perc_hard, max_perc_hard, min_perc_easy, max_perc_easy, random_seed, rep_index):
    min_perc_hard = float(min_perc_hard)
    max_perc_hard = float(max_perc_hard)
    min_perc_easy = float(min_perc_easy)
    max_perc_easy = float(max_perc_easy)
    random_seed = int(random_seed)
    rep_index = int(rep_index)

    random.seed(random_seed)
    with open(OUR_OUTPUT_FILENAME, 'rb') as f:
        outputs = pickle.load(f)

    hard_examples = select_examples(outputs, min_perc_hard, max_perc_hard)
    easy_examples = select_examples(outputs, min_perc_easy, max_perc_easy)
    query = form_query(hard_examples, easy_examples)
    f = open('log_LLM_tweaking_seed%d_rep%d.txt'%(random_seed, rep_index), 'w')
    print('QUERY:\n---------------------------------------')
    f.write('QUERY:\n---------------------------------------\n')
    print(query)
    f.write(query + '\n')
    reply = query_LLM(query)
    print('REPLY:\n----------------------------------------')
    f.write('REPLY:\n----------------------------------------')
    print(reply)
    f.write(reply + '\n')
    f.close()


def usage():
    print('Usage: python LLM_tweak_prompt.py <min_perc_hard> <max_perc_hard> <min_perc_easy> <max_perc_easy> <random_seed> <rep_index>')


if __name__ == '__main__':
    LLM_tweak_prompt(*(sys.argv[1:]))
