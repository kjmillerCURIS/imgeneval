import os
import sys
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from datasets import Dataset
from text_embedder_finetuning_dataset import TextEmbedderFinetuningDataset


if __name__ == '__main__':
    print('setting up model...')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    #train_dataset = Dataset.from_dict({
    #    "sentence1": ["It's nice weather outside today.", "He drove to work."],
    #    "sentence2": ["It's so sunny.", "She walked to the store."],
    #    "score": [1.0, 0.3],
    #})
    #print(train_dataset.column_names)
    #print(train_dataset.info)
    print('loading dataset...')
    train_dataset = TextEmbedderFinetuningDataset().to_huggingface()
    print('loss...')
    loss = losses.AnglELoss(model)

    print('setting up trainer...')
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        loss=loss,
    )
    print('training...')
    trainer.train()
    print('done!')
