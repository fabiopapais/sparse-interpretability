from typing import List
import torch
import numpy as np
from autoencoders.learned_dict import TiedSAE
from autoencoders.sae_ensemble import FunctionalTiedSAE
from transformers import AutoModel, AutoTokenizer
from delphi.config import CacheConfig, RunConfig
from delphi.latents import LatentCache
from delphi.sparse_coders import load_hooks_sparse_coders
from delphi.utils import load_tokenized_data
from pathlib import Path

device = torch.device('cuda:0')

SEQUENCE_LEN = 256
EMBEDDING_DIM = 512

activation_dim = SEQUENCE_LEN * EMBEDDING_DIM
ratio = 1.0
latent_dim = int(ratio * activation_dim)
model_weights_filename = '../sweep_outputs/learned_dicts_epoch_0_chunk_17.pt'

print(np.logspace(-4, -2, 16))
models: List[TiedSAE] = torch.load(model_weights_filename, weights_only=False)

model = models[-1]

def modelfunc(batch):
    with torch.no_grad():
        return model.encode(batch)

l1 = 0.01

hookpoint_to_sparse_encode = {
    'blocks.2.hook_resid_post': modelfunc
}

print(hookpoint_to_sparse_encode)

# 1. Configuração
# Supondo que você tenha uma RunConfig como a do __main__.py
# Aqui estão os parâmetros essenciais para o cache
model_name = "EleutherAI/pythia-70m" # Seu modelo base
sae_name = "EleutherAI/sae-pythia-70m-32k" # Seu SAE
cache_cfg = CacheConfig(
    dataset_repo="EleutherAI/SmolLM2-135M-10B",
    dataset_split="train[:1%]",
    n_tokens=10_000,
    batch_size=8,
    cache_ctx_len=256,
    n_splits=5 # Número de arquivos para salvar o cache
)

# Diretório para salvar os resultados
latents_path = Path("./results/my_run/latents")
latents_path.mkdir(parents=True, exist_ok=True)

# 2. Carregar Modelos
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Carrega o SAE e o acopla ao modelo base via hooks
# (Simplificação da lógica em __main__.py)
run_cfg = RunConfig(model=model_name, sparse_model=sae_name, hookpoints=hookpoints, cache_cfg=cache_cfg, constructor_cfg=None, sampler_cfg=None)

# hookpoint_to_sparse_encode, transcode = load_hooks_sparse_coders(model, run_cfg)
# hookpoint_to_sparse_encode uses custom logic

# 3. Preparar Dataset
tokens = load_tokenized_data(
    ctx_len=cache_cfg.cache_ctx_len,
    tokenizer=tokenizer,
    dataset_repo=cache_cfg.dataset_repo,
    dataset_split=cache_cfg.dataset_split,
)

# 4. Executar Caching
cache = LatentCache(
    model,
    hookpoint_to_sparse_encode,
    batch_size=cache_cfg.batch_size
)

# Inicia o processo de passar os dados pelo modelo e coletar ativações
cache.run(n_tokens=cache_cfg.n_tokens, tokens=tokens)

# 5. Salvar os resultados
cache.save_splits(
    n_splits=cache_cfg.n_splits,
    save_dir=latents_path
)

print("Caching concluído!")


