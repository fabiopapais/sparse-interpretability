from typing import List
import torch
import numpy as np
from autoencoders.learned_dict import TiedSAE
from autoencoders.sae_ensemble import FunctionalTiedSAE

device = torch.device('cpu')

SEQUENCE_LEN = 256
EMBEDDING_DIM = 512

activation_dim = SEQUENCE_LEN * EMBEDDING_DIM
ratio = 1.0
latent_dim = int(ratio * activation_dim)
model_weights_filename = './sweep_outputs/learned_dicts_epoch_0_chunk_17.pt'

print(np.logspace(-4, -2, 16))
models: List[TiedSAE] = torch.load(model_weights_filename, weights_only=False)

model = models[-1]

def modelfunc(batch):
    with torch.no_grad():
        return model.encode(batch)

l1 = 0.01

hookpoint_to_sparse_encode = {
    'layer2': modelfunc
}

print(hookpoint_to_sparse_encode)

