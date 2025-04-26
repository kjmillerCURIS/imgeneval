import os
import sys
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
from sentence_transformers.training_args import BatchSamplers
from text_embedder_finetuning_dataset import TextEmbedderFinetuningDataset
from train_vqa_policy import BASE_DIR


NUM_EPOCHS = 20
BATCH_SIZE = 16
SAVE_STEPS = 250
EMBEDDER_TYPE = 'all-MiniLM-L6-v2'


def finetune_text_embedder(loss_type, lr):
    lr = float(lr)

    print('loading model...')
    model = SentenceTransformer(EMBEDDER_TYPE, device='cuda')

    print('loading dataset...')
    train_dataset = TextEmbedderFinetuningDataset().to_huggingface()

    print('creating loss...')
    if loss_type == 'CoSENT':
        loss = losses.CoSENTLoss(model)
    elif loss_type == 'AnglE':
        loss = losses.AnglELoss(model)
    else:
        assert(False)

    print('creating train args...')
    output_dir = os.path.join(BASE_DIR, 'embedder_finetuning_%s_%s_lr%f'%(EMBEDDER_TYPE, loss_type, lr))
    os.makedirs(output_dir, exist_ok=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=lr, #default 2e-5
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        save_strategy='steps',
        save_steps=SAVE_STEPS,
        save_total_limit=None,
        logging_steps=SAVE_STEPS,
        run_name=os.path.basename(output_dir),  # Will be used in W&B if `wandb` is installed
    )

    print('creating trainer...')
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    print('training...')
    trainer.train()

    print('saving final checkpoint...')
    model.save_pretrained(os.path.join(output_dir, 'final'))


def usage():
    print('Usage: python finetune_text_embedder.py <loss_type> <lr>')


if __name__ == '__main__':
    finetune_text_embedder(*(sys.argv[1:]))
