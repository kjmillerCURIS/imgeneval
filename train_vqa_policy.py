import os
import sys
from collections import deque
import json
import numpy as np
from PIL import Image
import random
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print(f'hey lets make lots of fucking tools to do slightly different versions of the same thing ane make them all fucking incompatible with each other cuz were so fucking smart, anyway, "{e}"')

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


NUM_EPOCHS = 100
SAVE_FREQ = 10
LEARNING_RATE = 0.001
INPUT_DIM = 384 + 512
BUFFER_CAPACITY = 1000
BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_END = 0.1
BASE_DIR = os.path.expanduser('~/data/vislang-domain-exploration-data/imgeneval-data')
COCO_JSON_PATH = os.path.join(BASE_DIR, 'coco_image_data_final2_with_context_and_preds.json')
COCO_ENHANCED_JSON_PATH = os.path.splitext(COCO_JSON_PATH)[0] + '_enhanced_with_dispreferred_prompt_questions.json'


class CustomDataset:
    def __init__(self, data, batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self.index = 0  # Iterator index

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.data):
            raise StopIteration
        batch = self.data[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return batch

    def __len__(self):
        """Returns the number of batches"""
        return (len(self.data) + self.batch_size - 1) // self.batch_size  # Round up

    def sample(self):
        """Returns a single random sample"""
        return random.choice(self.data) if self.data else None


class VanillaSupervisionDataset(Dataset):

    def __init__(self, dataset, include_dispreferred_prompt_questions):
        self.contexts = []
        self.rewards = []
        for batch in dataset:
            for sample in batch:
                for question_obj in sample['questions']:
                    context = question_obj['context']
                    reward = [question_obj['blip_reward'], question_obj['vilt_reward']]
                    self.contexts.append(context)
                    self.rewards.append(reward)

                if include_dispreferred_prompt_questions:
                    for question_obj in sample['questions']:
                        context = question_obj['context']
                        reward = [question_obj['blip_reward'], question_obj['vilt_reward']]
                        self.contexts.append(context)
                        self.rewards.append(reward)

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return {'context' : torch.FloatTensor(self.contexts[idx]), 'reward' : torch.FloatTensor(self.rewards[idx])}


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, action_dim):
        super(QNetwork, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU()]
        for i in range(1, len(hidden_dims)):
            layers.extend([nn.Linear(hidden_dims[i-1], hidden_dims[i]), nn.ReLU()])

        layers.append(nn.Linear(hidden_dims[-1], action_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward):
        self.buffer.append((state, action, reward))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards)

    def __len__(self):
        return len(self.buffer)


def answer_with_blip(image: Image.Image, question: str, processor_blip, model_blip, device) -> str:
    """Answer a VQA question using BLIP with a PIL Image"""
    try:
        inputs = processor_blip(image, question, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model_blip.generate(**inputs)
        return processor_blip.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        print(f"BLIP error: {e}")
        return "error"


def answer_with_vilt(image: Image.Image, question: str, processor_vilt, model_vilt, device) -> str:
    """Answer a VQA question using VILT with a PIL Image"""
    try:
        encoding = processor_vilt(image, question, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model_vilt(**encoding)
        return model_vilt.config.id2label[outputs.logits.argmax(-1).item()]
    except Exception as e:
        print(f"VILT error: {e}")
        return "error"


def answer_with_ofa_generic(image: Image.Image, question: str, patch_transform_ofa, tokenizer_ofa, model_ofa, device) -> str:
    patch_image = patch_transform_ofa(image).unsqueeze(0).to(device)
    inputs = tokenizer_ofa([question], return_tensors="pt").input_ids.to(device)
    gen = model_ofa.generate(inputs, patch_images=patch_image, num_beams=5, no_repeat_ngram_size=3)
    return tokenizer_ofa.batch_decode(gen, skip_special_tokens=True)


def answer_with_ofabase(image: Image.Image, question: str, patch_transform_ofabase, tokenizer_ofabase, model_ofabase, device) -> str:
    try:
        answer_with_ofa_generic(image, question, patch_transform_ofabase, tokenizer_ofabase, model_ofabase, device)
    except Exception as e:
        print(f"OFA-Base error: {e}")
        return "error"


def answer_with_ofalarge(image: Image.Image, question: str, patch_transform_ofalarge, tokenizer_ofalarge, model_ofalarge, device) -> str:
    try:
        answer_with_ofa_generic(image, question, patch_transform_ofalarge, tokenizer_ofalarge, model_ofalarge, device)
    except Exception as e:
        print(f"OFA-Large error: {e}")
        return "error"


# Action Space Configuration
ACTION_SPACE = {
    0: {"name": "Blip", "function": answer_with_blip},
    1: {"name": "Vilt", "function": answer_with_vilt},
    2: {"name": "ofabase", "function": answer_with_ofabase},
    3: {"name": "ofalarge", "function": answer_with_ofalarge},
}
ACTION_DIM = len(ACTION_SPACE)


# Context Builder with proper image handling
class ContextBuilder:
    def __init__(self, device='cuda'):
        self.device=device
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def build(self, image: Image.Image, question: str) -> np.ndarray:
        """Build context vector from image and question"""

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


def load_mscoco_samples(json_path: str = COCO_ENHANCED_JSON_PATH):
    with open(json_path, 'r') as file:
        data = json.load(file)

    return CustomDataset(data)


def train_vqa_policy_RL(hidden_dims, include_dispreferred_prompt_questions):

    device = 'cpu'

    # Initialize components
    q_net = QNetwork(INPUT_DIM, hidden_dims, ACTION_DIM).to(device)
    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    buffer = ReplayBuffer(BUFFER_CAPACITY)
    epsilon = EPSILON_START

    # Load dataset
    train_dataset = load_mscoco_samples()
    for epoch in range(NUM_EPOCHS):
        for batch in train_dataset:
            for sample in batch:
                img_url, questions = sample['coco_url'], sample['questions']
                if include_dispreferred_prompt_questions:
                    questions += sample['dispreferred_prompt_questions']

                for question_obj in questions:
                    question = question_obj['question']
                    ground_truth = question_obj['answer']

                    # Build context
                    context = question_obj['context']

                    if np.random.rand() < epsilon:
                        action = np.random.randint(ACTION_DIM)
                    else:
                        with torch.no_grad():
                            q_values = q_net(torch.FloatTensor(context).to(device))
                            action = torch.argmax(q_values).item()

                    # Check if both BLIP and VILT failed (rewards == -1) and override action
                    '''
                    if question_obj['blip_reward'] == -1 and question_obj['vilt_reward'] == -1:
                        action = 2
                    '''


                    # Get prediction
                    try:
                        reward = 0
                        if action == 0:
                            reward = question_obj['blip_reward']
                            predicted = question_obj['blip_pred']
                        elif action == 1:
                            reward = question_obj['vilt_reward']
                            predicted = question_obj['vilt_pred']

                        elif action == 2:
                            reward = 1
                            predicted = -2

                        #print(action)
                        #print(reward)
                    except KeyError:
                        print(f"Invalid action {action}")
                        reward = -1
                    except Exception as e:
                        print(f"Prediction error: {e}")
                        reward = -1

                    # Store in replay buffer
                    try:
                        buffer.add(context, action, reward)
                    except ValueError as e:
                        print(f"Buffer error: {e}")
                        continue

            if len(buffer) >= BATCH_SIZE:
                contexts, actions, rewards = buffer.sample(BATCH_SIZE)
                contexts = torch.FloatTensor(contexts).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                #print(actions)
                #print(rewards)
                q_values = q_net(contexts)
                #print(q_values)
                q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()
                #print(q_selected)
                loss = nn.MSELoss()(q_selected, rewards)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        epsilon = max(EPSILON_END, epsilon * 0.995)
        print(f"Epsilon: {epsilon}")

        # Save model after each epoch
        if epoch % SAVE_FREQ == 0:
            hidden_dims_str = '_'.join([str(h) for h in hidden_dims])
            model_prefix = f"trained_q_network_RL_hidden_dims_{hidden_dims_str}"
            if include_dispreferred_prompt_questions:
                model_prefix += "_trainwithdispreferredquestions"

            model_suffix = f"_epoch_{epoch}.pth"
            model_dir = os.path.join(BASE_DIR, model_prefix)
            os.makedirs(model_dir, exist_ok=True)
            model_save_path = os.path.join(model_dir, model_prefix + model_suffix)
            torch.save({
                    'model_state_dict': q_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'input_dim': INPUT_DIM,
                    'hidden_dims': hidden_dims,
                    'action_dim': ACTION_DIM,
                }, model_save_path)
            print(f"Model saved to {model_save_path}")


def train_vqa_policy_vanilla_supervision(hidden_dims, include_dispreferred_prompt_questions):

    device = 'cpu'

    # Initialize components
    q_net = QNetwork(INPUT_DIM, hidden_dims, ACTION_DIM).to(device)
    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)

    # Load dataset
    train_dataset = load_mscoco_samples()
    train_dataset = VanillaSupervisionDataset(train_dataset, include_dispreferred_prompt_questions)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(NUM_EPOCHS):
        for batch in train_dataloader:
            context, reward = batch['context'].to(device), batch['reward'].to(device)
            q_selected = q_net(context)
            loss = nn.MSELoss(reduction='sum')(q_selected, reward) / context.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


        # Save model after each epoch
        if epoch % SAVE_FREQ == 0:
            hidden_dims_str = '_'.join([str(h) for h in hidden_dims])
            model_prefix = f"trained_q_network_vanilla_supervision_hidden_dims_{hidden_dims_str}"
            if include_dispreferred_prompt_questions:
                model_prefix += "_trainwithdispreferredquestions"

            model_suffix = f"_epoch_{epoch}.pth"
            model_dir = os.path.join(BASE_DIR, model_prefix)
            os.makedirs(model_dir, exist_ok=True)
            model_save_path = os.path.join(model_dir, model_prefix + model_suffix)
            torch.save({
                    'model_state_dict': q_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'input_dim': INPUT_DIM,
                    'hidden_dims': hidden_dims,
                    'action_dim': ACTION_DIM,
                }, model_save_path)
            print(f"Model saved to {model_save_path}")


def train_vqa_policy(learning_type, hidden_dims, include_dispreferred_prompt_questions):
    hidden_dims = [int(h) for h in hidden_dims.split('_')]
    include_dispreferred_prompt_questions = int(include_dispreferred_prompt_questions)
    if learning_type == 'RL':
        train_vqa_policy_RL(hidden_dims, include_dispreferred_prompt_questions)
    elif learning_type == 'vanilla_supervision':
        train_vqa_policy_vanilla_supervision(hidden_dims, include_dispreferred_prompt_questions)
    else:
        assert(False)


if __name__ == '__main__':
    train_vqa_policy(*(sys.argv[1:]))
