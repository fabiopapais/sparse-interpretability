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

print("Started Loading the Model and the dataset")
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device='cuda:0')
dataset = datasets.load_dataset("Salesforce/wikitext", name="wikitext-2-raw-v1", split="train")
print("Finished Loading model and dataset")

# --- Configurações ---
SAVE_CHUNK_SIZE = 200  # Salvar um arquivo a cada 200 batches
BATCH_SIZE = 8
OUTPUT_DIR = "./activations_chunks"
LAYER = 2

# Criar o diretório de saída se não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQUENCE_LENGTH = 100

def streaming_sequence_generator(dataset, sequence_length, tokenizer):
    """
    Generator that yields sequences of exactly sequence_length tokens
    by streaming through the dataset and concatenating samples.
    """
    eos_token = tokenizer.eos_token or "<|endoftext|>"
    token_buffer = []

    for item in dataset:
        # Skip empty texts
        if not item["text"].strip():
            continue

        # Tokenize the current text and add EOS
        text_with_eos = eos_token + item["text"]
        tokens = tokenizer(text_with_eos, add_special_tokens=False)['input_ids']

        # Add tokens to buffer
        token_buffer.extend(tokens)

        # Extract sequences while we have enough tokens
        while len(token_buffer) >= sequence_length:
            # Extract exactly sequence_length tokens
            sequence = token_buffer[:sequence_length]
            # Keep remaining tokens in buffer
            token_buffer = token_buffer[sequence_length:]

            yield torch.tensor(sequence, dtype=torch.long)

def create_batches_from_sequences(sequence_generator, batch_size):
    """
    Group individual sequences into batches.
    """
    batch = []
    for sequence in sequence_generator:
        batch.append(sequence)

        if len(batch) == batch_size:
            # Convert list of sequences to batch tensor
            batch_tensor = torch.stack(batch)
            yield batch_tensor
            batch = []

    # Yield remaining sequences if any (partial batch)
    if batch:
        batch_tensor = torch.stack(batch)
        yield batch_tensor

print("Full dataset data generators")
# Create sequence generator and batch generator
sequence_gen = streaming_sequence_generator(dataset, SEQUENCE_LENGTH, model.tokenizer)
batch_gen = create_batches_from_sequences(sequence_gen, BATCH_SIZE)
print("Full dataset data generator done")

# --- Loop de Coleta e Salvamento em Chunks ---
activations_buffer = torch.Tensor().to('cuda:0')
chunk_index = 0

print(f"Coletando ativações e salvando em chunks a cada {SAVE_CHUNK_SIZE} batches...")
for i, tokens_batch in enumerate(tqdm(batch_gen, desc="Processing batches")):
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