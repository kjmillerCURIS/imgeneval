import os
import sys
import numpy as np
import pickle
import random
import torch
from tqdm import tqdm
print('importing open-clip...')
import open_clip
print('done importing open-clip')
from sentence_transformers import SentenceTransformer
from transformers import CLIPTokenizer, CLIPModel
from winoground_inference_with_LLM_fusion import load_winoground_data, BASE_DIR


DEVICE = 'cuda'
NEGCLIP_PATH = 'negCLIP.pt'
NEGCLIP_BACKBONE = 'ViT-B-32'


#use this in case we have to do it in chunks
def compute_text_embeddings_helper(captions, embedder_type, tools):
    if embedder_type == 'clip':
        inputs = tools['clip']['tokenizer'](captions, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            return tools['clip']['model'].get_text_features(**inputs).cpu().numpy()
    elif embedder_type == 'negclip':
        tokens = open_clip.tokenize(captions).to(DEVICE)
        with torch.no_grad():
            return tools['negclip']['model'].encode_text(tokens).cpu().numpy()
    else:
        return tools[embedder_type].encode(captions)


#captions should be a list
#will return (N, *) tensor where N is number of captions
def compute_text_embeddings(captions, embedder_type, tools):
    #just try doing 'em all for now
    return compute_text_embeddings_helper(captions, embedder_type, tools)


#return embedder_types, tools
def setup_tools():
    tools = {}
    for thingy in ['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2', 'all-distilroberta-v1', 'all-mpnet-base-v2', 'multi-qa-mpnet-base-dot-v1', 'multi-qa-distilbert-cos-v1', 'paraphrase-albert-small-v2', 'multi-qa-MiniLM-L6-cos-v1', 'paraphrase-MiniLM-L3-v2']:
        tools[thingy] = SentenceTransformer(thingy, device=DEVICE)

    clip_model_name = 'openai/clip-vit-base-patch32'
    tools['clip'] = {'tokenizer' : CLIPTokenizer.from_pretrained(clip_model_name), 'model' : CLIPModel.from_pretrained(clip_model_name)}
    tools['clip']['model'].to(DEVICE)
    tools['clip']['model'].eval()
    tools['negclip'] = {}
    tools['negclip']['model'], _, __ = open_clip.create_model_and_transforms('ViT-B-32', pretrained=NEGCLIP_PATH, device=DEVICE, load_weights_only=False)
    return sorted(tools.keys()), tools


def measure_text_embedders_on_winoground_captions():
    embedder_types, tools = setup_tools()
    data = load_winoground_data()
    captions_A = [example[0][0]['prompt'] for example in data]
    captions_B = [example[1][0]['prompt'] for example in data]
    stats = {}
    for embedder_type in tqdm(embedder_types):
        print(embedder_type)
        embeddings_A = compute_text_embeddings(captions_A, embedder_type, tools)
        embeddings_B = compute_text_embeddings(captions_B, embedder_type, tools)
        embeddings_A = embeddings_A / np.linalg.norm(embeddings_A, axis=-1, keepdims=True)
        embeddings_B = embeddings_B / np.linalg.norm(embeddings_B, axis=-1, keepdims=True)
        cossims = embeddings_A @ embeddings_B.T
        pos_cossims, neg_cossims = [], []
        for i in range(cossims.shape[0]):
            for j in range(cossims.shape[1]):
                if i == j:
                    pos_cossims.append(cossims[i,j])
                else:
                    neg_cossims.append(cossims[i,j])

        pos_mean = np.mean(pos_cossims)
        pos_sd = np.std(pos_cossims, ddof=1)
        neg_mean = np.mean(neg_cossims)
        neg_sd = np.std(neg_cossims, ddof=1)
        stats[embedder_type] = {'pos' : {'mean' : pos_mean, 'sd' : pos_sd}, 'neg' : {'mean' : neg_mean, 'sd' : neg_sd}}
        print(embedder_type)
        print(stats[embedder_type])

    with open(os.path.join(BASE_DIR, 'winoground_caption_cossim_stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)

    f = open(os.path.join(BASE_DIR, 'winoground_caption_cossim_stats.csv'), 'w')
    f.write('embedding model,mean cossim (paired),SD cossim (paired),mean cossim (unpaired),SD cossim (unpaired)\n')
    for embedder_type in embedder_types:
        f.write(','.join([embedder_type,str(stats[embedder_type]['pos']['mean']),str(stats[embedder_type]['pos']['sd']),str(stats[embedder_type]['neg']['mean']),str(stats[embedder_type]['neg']['sd'])]) + '\n')

    f.close()

    print(stats)


if __name__ == '__main__':
    measure_text_embedders_on_winoground_captions()
