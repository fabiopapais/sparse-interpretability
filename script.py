# This file contains the script to run the model and extract activations from the 44k rows version of the dataset,
# IT IS NOT UP TO DATE WITH THE ipynb!!!!
# It extracts the activations and saves them on a file, and is useful as a script to run in a cluster

from transformer_lens import HookedTransformer
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import datasets
import os

torch.manual_seed(42)

model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device='cuda:0')
dataset = datasets.load_dataset("Salesforce/wikitext", name="wikitext-2-raw-v1", split="train[:30000]")

# --- Configurações ---
SAVE_CHUNK_SIZE = 200  # Salvar um arquivo a cada 200 batches
BATCH_SIZE = 8
OUTPUT_DIR = "./activations_chunks"
LAYER = 2
 
# Criar o diretório de saída se não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
def create_batch(batch):
    """
    Create a batch where each sequence has exactly SEQUENCE_LENGTH tokens.
    Pad sequences with text from subsequent sequences instead of EOS tokens.
    """

    texts = [item["text"] for item in batch]
    eos_token = model.tokenizer.eos_token or "<|endoftext|>"
    
    # Create a continuous stream of text by joining all texts with EOS
    continuous_text = eos_token.join([""] + texts)
    
    # Tokenize the continuous text
    tokens = model.tokenizer(continuous_text, add_special_tokens=False)['input_ids']
    
    # Create sequences of exactly SEQUENCE_LENGTH tokens
    sequences = []
    for i in range(BATCH_SIZE):
        start_idx = i * 256
        end_idx = start_idx + 256
        
        if end_idx <= len(tokens):
            # We have enough tokens for this sequence
            sequence = tokens[start_idx:end_idx]
        else:
            # Not enough tokens, need to pad with more text
            # This should rarely happen if BATCH_SIZE and dataset are properly sized
            sequence = tokens[start_idx:]
            # Pad with beginning of the token stream if needed
            while len(sequence) < 256:
                remaining = 256 - len(sequence)
                pad_tokens = tokens[:min(remaining, len(tokens))]
                sequence.extend(pad_tokens)
            sequence = sequence[:256]  # Ensure exact length
        
        sequences.append(sequence)
    
    # Convert to tensor
    batch_tensor = torch.tensor(sequences, dtype=torch.long)
    return batch_tensor
 
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=create_batch, drop_last=True)
 
# --- Loop de Coleta e Salvamento em Chunks ---
activations_buffer = torch.Tensor().to('cuda:0')
chunk_index = 0
 
print(f"Coletando ativações e salvando em chunks a cada {SAVE_CHUNK_SIZE} batches...")
for i, tokens_batch in enumerate(tqdm(dataloader)):
    if tokens_batch is None:
        continue
    
    tokens_batch = tokens_batch.to('cuda:0')
    _, cache = model.run_with_cache(tokens_batch, remove_batch_dim=False)
 
    # Adiciona as ativações do batch atual ao buffer na CPU
    key = f"blocks.{LAYER}.hook_resid_post"
    activations_buffer = torch.cat((activations_buffer, cache[key].cuda()), 0)

    del tokens_batch, cache, _
 
    # Verifica se é hora de salvar um chunk
    if (i + 1) % SAVE_CHUNK_SIZE == 0:
        file_path = os.path.join(OUTPUT_DIR, f"layer_{LAYER}_chunk_{chunk_index}.pt")
        torch.save(activations_buffer, file_path)
        print(activations_buffer.shape)
 
        # Limpa o buffer para o próximo chunk e atualiza o índice
        activations_buffer = torch.Tensor().to('cuda:0')
        chunk_index += 1
 
# # --- Salva qualquer resto que sobrou no buffer ---
# print("Salvando o chunk final...")
# for layer in layers:
#     if activations_buffer[layer]: # Se houver algo no buffer
#         chunk_tensor = torch.cat(activations_buffer[layer], dim=0)
#         file_path = os.path.join(OUTPUT_DIR, f"layer_{layer}_chunk_{chunk_index}.pt")
#         torch.save(chunk_tensor, file_path)
 
print(f"Concluído! Ativações salvas em múltiplos arquivos no diretório '{OUTPUT_DIR}'")

