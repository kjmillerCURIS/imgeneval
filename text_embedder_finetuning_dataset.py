import os
import sys
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import Dataset as HuggingFaceDataset
from generate_data_for_finetuning_text_embedder import get_output_filename, DATA_STRIDE


class TextEmbedderFinetuningDataset(Dataset):
    def __init__(self):
        data = []
        for offset in tqdm(range(DATA_STRIDE)):
            filename = get_output_filename(offset)
            with open(filename, 'rb') as f:
                data_chunk = pickle.load(f)

            data.extend(data_chunk)

        self.positive_pairs = []
        self.negative_pairs = []
        for datum in tqdm(data):
            for positive_example in datum['positive_examples']:
                self.positive_pairs.append((datum['caption'], positive_example))

            for negative_example in datum['negative_examples']:
                self.negative_pairs.append((datum['caption'], negative_example))

    def __len__(self):
        return len(self.positive_pairs) + len(self.negative_pairs)

    def __getitem__(self, idx):
        if idx < len(self.positive_pairs):
            return {'sentence1' : self.positive_pairs[idx][0], 'sentence2' : self.positive_pairs[idx][1], 'score' : 1.0}
        else:
            idx -= len(self.positive_pairs)
            return {'sentence1' : self.negative_pairs[idx][0], 'sentence2' : self.negative_pairs[idx][1], 'score' : 0.0}

    def to_huggingface(self):
        d = {'sentence1' : [], 'sentence2' : [], 'score' : []}
        for idx in tqdm(range(len(self))):
            datum = self[idx]
            for k in ['sentence1', 'sentence2', 'score']:
                d[k].append(datum[k])

        return HuggingFaceDataset.from_dict(d)
