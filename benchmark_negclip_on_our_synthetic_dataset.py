import os
import sys
import json
import numpy as np
import pickle
from matplotlib import pyplot as plt
import random
import torch
from tqdm import tqdm
import open_clip
from train_vqa_policy import BASE_DIR, COCO_JSON_PATH
from generate_data_for_finetuning_text_embedder import get_output_filename as get_caption_data_filename
from generate_data_for_finetuning_text_embedder import DATA_STRIDE as CAPTION_DATA_STRIDE
from enhance_coco_with_dispreferred_prompt_questions import load_image_from_url
from measure_text_embedders_on_winoground_captions import NEGCLIP_PATH, NEGCLIP_BACKBONE



#each entry in output should have anchor, T+, T-, img_id, cossim(img, T+), cossim(img, T-)
#also should plot histogram of cossim(img, T-) - cossim(img, T+)


RANDOM_SEED = 0
OUR_OUTPUT_FILENAME = os.path.join(BASE_DIR, 'our_synthetic_data_with_negclip.pkl')
OUR_PLOT_FILENAME = os.path.join(BASE_DIR, 'our_synthetic_data_with_negclip.png')


def load_caption_data():
    data_list = []
    for offset in range(CAPTION_DATA_STRIDE):
        caption_data_filename = get_caption_data_filename(offset)
        with open(caption_data_filename, 'rb') as f:
            data_list.append(pickle.load(f))

    indices = [0 for _ in range(len(data_list))]
    out_data = []
    t = 0
    while True:
        offset = t % CAPTION_DATA_STRIDE
        if indices[offset] >= len(data_list[offset]):
            assert(all([index == len(data) for index, data in zip(indices, data_list)]))
            break

        out_data.append(data_list[offset][indices[offset]])
        indices[offset] += 1
        t += 1

    return out_data


def load_image_data():
    with open(COCO_JSON_PATH, 'r') as f:
        data = json.load(f)

    return data


def setup_tools():
    negclip_model, _, negclip_preprocess = open_clip.create_model_and_transforms(NEGCLIP_BACKBONE, pretrained=NEGCLIP_PATH, device='cuda', load_weights_only=False)
    negclip_model.eval()
    negclip_tokenizer = open_clip.get_tokenizer(NEGCLIP_BACKBONE)
    return {'negclip_model' : negclip_model, 'negclip_preprocess' : negclip_preprocess, 'negclip_tokenizer' : negclip_tokenizer}


#returns a single NORMALIZED embedding
def embed_image(image_url, tools):
    image = load_image_from_url(image_url)
    image = tools['negclip_preprocess'](image).to('cuda').unsqueeze(0)
    with torch.no_grad():
        image_features = tools['negclip_model'].encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    return image_features


#returns an array/tensor of NORMALIZED embeddings
def embed_texts(texts, tools):
    texts = tools['negclip_tokenizer'](texts).to('cuda')
    with torch.no_grad():
        text_features = tools['negclip_model'].encode_text(texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features


#return a list of dicts, each containing anchor, T+, T-, image_url, cossim(img, T+), cossim(img, T-)
def process_one_item(caption_item, image_url, tools):
    num_examples = min(len(caption_item['positive_examples']), len(caption_item['negative_examples']))
    if num_examples == 0:
        print('!!!')
        return []

    positive_examples = random.sample(caption_item['positive_examples'], num_examples)
    negative_examples = random.sample(caption_item['negative_examples'], num_examples)
    outputs = []
    image_embedding = embed_image(image_url, tools)
    positive_embeddings = embed_texts(positive_examples, tools)
    negative_embeddings = embed_texts(negative_examples, tools)
    with torch.no_grad():
        positive_cossims = (positive_embeddings @ image_embedding.t()).squeeze(-1).cpu().numpy()
        negative_cossims = (negative_embeddings @ image_embedding.t()).squeeze(-1).cpu().numpy()

    for positive_example, negative_example, positive_cossim, negative_cossim in zip(positive_examples, negative_examples, positive_cossims, negative_cossims):
        outputs.append({'caption' : caption_item['caption'], 'image_url' : image_url, 'positive_example' : positive_example, 'negative_example' : negative_example, 'positive_cossim' : positive_cossim, 'negative_cossim' : negative_cossim})

#    print(min([output['negative_cossim'] - output['positive_cossim'] for output in outputs]))
#    print(max([output['negative_cossim'] - output['positive_cossim'] for output in outputs]))
    return outputs


def make_plot(outputs):
    vals = [output['negative_cossim'] - output['positive_cossim'] for output in outputs]
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.set_title('Histogram and CDF of cossim(img,T-) - cossim(img,T+)')
    ax1.set_xlabel('cossim(img,T-) - cossim(img,T+)')
    ax1.set_ylabel('Count')
    counts, bins, patches = ax1.hist(vals, bins=30, color='skyblue', alpha=0.6)
    ax2 = ax1.twinx()
    ax2.set_ylabel('CDF probability')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    cdf = np.cumsum(counts)
    cdf = cdf / cdf[-1]
    ax2.plot(bin_centers, cdf, color='darkred')
    plt.tight_layout()
    plt.savefig(OUR_PLOT_FILENAME)
    plt.clf()


def benchmark_negclip_on_our_synthetic_dataset():
    random.seed(RANDOM_SEED)
    caption_data = load_caption_data()
    image_data = load_image_data()
    tools = setup_tools()
    outputs = []
    for caption_datum, image_datum in tqdm(zip(caption_data, image_data)):
        assert(caption_datum['caption'] == image_datum['prompt'])
        outputs.extend(process_one_item(caption_datum, image_datum['coco_url'], tools))

    with open(OUR_OUTPUT_FILENAME, 'wb') as f:
        pickle.dump(outputs, f)

    make_plot(outputs)


def usage():
    print('Usage: python benchmark_negclip_on_our_synthetic_dataset.py')


if __name__ == '__main__':
    benchmark_negclip_on_our_synthetic_dataset()
