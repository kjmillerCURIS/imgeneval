import os
import sys
import json
import pickle
import re
from tqdm import tqdm
from train_vqa_policy import BASE_DIR, COCO_JSON_PATH
from enhance_coco_with_dispreferred_prompt_questions import generate_neg_prompt
from winoground_inference_with_LLM_fusion import ask_deepseek


NUM_EXAMPLES_PER_CAPTION = 10
DATA_STRIDE = 6
SAVE_FREQ = 5


def get_output_filename(offset):
    return os.path.join(BASE_DIR, 'data_for_finetuning_text_embedder_%d.pkl'%(offset))


def load_captions(offset):
    with open(COCO_JSON_PATH, 'r') as f:
        data = json.load(f)

    captions = [datum['prompt'] for datum in data]
    return captions[offset::DATA_STRIDE]


def generate_positive_example(caption):
    prompt_to_generate_description = f'''
        Given an input sentence describing a scene,
        please give a new sentence that has the exact same meaning but is phrased differently.
        For example, "a cat chasing a dog and a mouse" could be rephrased as "a cat chasing a mouse and a dog", "a dog and a mouse being chased by a cat", "a cat running after a dog and mouse", "a kitty chasing a doggy and a mouse", etc.
        Please give one possible rephrasing.

        Indicate your final response as:
        #Final Sentence#: Your final response here.

        The input is:{caption}
                                      '''

    result = ask_deepseek(prompt_to_generate_description)
    result = result.strip()
    match = re.search(r"#Final Sentence#:\s*(.+)", result)
    result = match.group(1) if match else None
    if result is None or 'final sentence' in result.lower():
        return None

    return result


def generate_negative_example(caption):
    result = generate_neg_prompt(caption)
    match = re.search(r"#Final Sentence#:\s*(.+)", result)
    result = match.group(1) if match else None
    if result is None or 'final sentence' in result.lower():
        return None

    return result


def generate_data_for_finetuning_text_embedder(offset):
    offset = int(offset)

    captions = load_captions(offset)
    output_filename = get_output_filename(offset)
    output = []
    start_index = 0
    if os.path.exists(output_filename):
        with open(output_filename, 'rb') as f:
            output = pickle.load(f)

        start_index = len(output)
        print('oh goody, already processed %d captions!'%(len(output)))

    for caption in tqdm(captions[start_index:]):
        positive_examples = []
        for _ in range(NUM_EXAMPLES_PER_CAPTION):
            positive_example = generate_positive_example(caption)
            if positive_example is None:
                print('!')
                continue

            if positive_example in positive_examples:
                print('?')
                continue

            positive_examples.append(positive_example)

        negative_examples = []
        for _ in range(NUM_EXAMPLES_PER_CAPTION):
            negative_example = generate_negative_example(caption)
            if negative_example is None:
                print('!!')
                continue

            if negative_example in negative_examples:
                print('??')
                continue

            negative_examples.append(negative_example)

        o = {'caption' : caption, 'positive_examples' : positive_examples, 'negative_examples' : negative_examples}
        print(o)
        output.append(o)
        if len(output) % SAVE_FREQ == 0 and len(output) > 0:
            with open(output_filename, 'wb') as f:
                pickle.dump(output, f)

    with open(output_filename, 'wb') as f:
        pickle.dump(output, f)


def usage():
    print('Usage: python generate_data_for_finetuning_text_embedder.py <offset>')


if __name__ == '__main__':
    generate_data_for_finetuning_text_embedder(*(sys.argv[1:]))
