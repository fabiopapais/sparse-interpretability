# import transformer_lens
# import transformer_lens.utils as utils
# from transformer_lens.hook_points import (
    # HookPoint,
# )  # Hooking utilities
from transformer_lens import HookedTransformer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import datasets
import os

torch.manual_seed(42)

model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device='cuda:0')

text = ["Hello, world!!!!!!!!!!!!!!, Hello, world!!!!!!!!!!!!!! Hello, world!!!!!!!!!!!!!!", "Hello,"]
tokens = model.to_tokens(text)
print(tokens)

logits, cache = model.run_with_cache(tokens)

print("Cache keys:", cache.keys())

dataset = datasets.load_dataset("Salesforce/wikitext", name="wikitext-2-raw-v1", split="train[:30000]", ignore_verifications=True)

# --- Configurações ---
SAVE_CHUNK_SIZE = 200  # Salvar um arquivo a cada 200 batches
BATCH_SIZE = 4
OUTPUT_DIR = "./activations_chunks"
 
# Criar o diretório de saída se não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
# (Assumindo que model, dataset, etc., estão definidos)
# ...

def create_batch(batch):
    texts = [item["text"] for item in batch if item['text']]
    if not texts:
        return None
    tokens = model.to_tokens(texts)
    return tokens

 
layers = list(range(1, 2))
# Não vamos mais usar um dicionário para guardar tudo, vamos salvar direto.
 
# Use drop_last=True para garantir que todos os chunks tenham o mesmo tamanho (exceto o último)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=create_batch, drop_last=True)
 
# --- Loop de Coleta e Salvamento em Chunks ---
activations_buffer = {layer: [] for layer in layers}
chunk_index = 0
 
print(f"Coletando ativações e salvando em chunks a cada {SAVE_CHUNK_SIZE} batches...")
for i, tokens_batch in enumerate(tqdm(dataloader)):
    if tokens_batch is None:
        continue
    
    tokens_batch = tokens_batch.to('cuda:0')
    _, cache = model.run_with_cache(tokens_batch, remove_batch_dim=False)
 
    # Adiciona as ativações do batch atual ao buffer na CPU
    for layer in layers:
        key = f"blocks.{layer}.hook_resid_post"
        activations_buffer[layer].append(cache[key].cpu())
 
    del tokens_batch, cache, _
 
    # Verifica se é hora de salvar um chunk
    if (i + 1) % SAVE_CHUNK_SIZE == 0:
        for layer in layers:
            # Concatena apenas o chunk atual
            chunk_tensor = torch.cat(activations_buffer[layer], dim=0)
            
            # Salva o chunk em um arquivo separado
            file_path = os.path.join(OUTPUT_DIR, f"layer_{layer}_chunk_{chunk_index}.pt")
            torch.save(chunk_tensor, file_path)
 
        # Limpa o buffer para o próximo chunk e atualiza o índice
        activations_buffer = {layer: [] for layer in layers}
        chunk_index += 1
 
# --- Salva qualquer resto que sobrou no buffer ---
print("Salvando o chunk final...")
for layer in layers:
    if activations_buffer[layer]: # Se houver algo no buffer
        chunk_tensor = torch.cat(activations_buffer[layer], dim=0)
        file_path = os.path.join(OUTPUT_DIR, f"layer_{layer}_chunk_{chunk_index}.pt")
        torch.save(chunk_tensor, file_path)
 
print(f"Concluído! Ativações salvas em múltiplos arquivos no diretório '{OUTPUT_DIR}'")

