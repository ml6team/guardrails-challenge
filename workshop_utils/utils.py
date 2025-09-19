import os
from typing import Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm



def load_hidden_states(topic: str,
                       device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.Tensor:
    
    BASE_PATH = Path(__file__).parent
    EMBED_DIM = 4096
    hidden_path = BASE_PATH / 'hidden_states' / topic
    hidden_states = torch.ones((1000, EMBED_DIM))

    ptr = 0
    for file_name in tqdm(os.listdir(hidden_path), desc='Loading tensor batches...'):
        if file_name.endswith('.pt') and 'batch' in file_name:
            hidden_batch = torch.load(os.path.join(hidden_path, file_name), map_location='cpu')
            batch_size = hidden_batch.shape[0]
            hidden_states[ptr:ptr+batch_size] = hidden_batch
            ptr += batch_size

    hidden_states = hidden_states[:ptr]
    hidden_states = hidden_states.to(dtype=torch.float32, device=device)

    return hidden_states


def load_representation_set(split: str,
                            device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu') -> dict:
    hidden_states = {}
    labels = {}

    prompt_types = [
    (f'valid_{split.lower()}', f'{split} Valid'),
    (f'invalid_{split.lower()}', f'{split} Invalid')
    ]

    for i, (type, prompt_name) in enumerate(prompt_types):
        X = load_hidden_states(topic=type,
                               device=device)
        hidden_states[type] = X
        labels[type] = i * torch.ones(X.shape[0], dtype=torch.long, device=device)

    print("Number of samples per prompt representation:")

    for type, prompt_name in prompt_types:
        print(f"  - {prompt_name}: {hidden_states[type].shape[0]}")

    print("\n")

    return hidden_states, labels