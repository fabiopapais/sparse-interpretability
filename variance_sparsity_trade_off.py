import torch
from torch import nn
from torch.nn import functional as F
import os
import numpy as np
import matplotlib.pyplot as plt

from autoencoder import TiedAutoencoder, train_autoencoder

epochs_sweep = list(range(2, 28, 2))
# epochs_sweep = [1, 2, 3]
sparsity_sweep = np.logspace(-5, -2, 20) # 20 alpha values from 1e-5 to 1e-2
# sparsity_sweep = np.logspace(-5, -2, 2)

print("Training with epochs_sweep:", epochs_sweep)
print("Training with sparsity_sweep:", sparsity_sweep)

dataset_dir = './activations_chunks'
R = 4
batch_size = 512
learning_rate = 1e-3

results = [[(-1, -1) for _ in sparsity_sweep] for _ in epochs_sweep]
print(len(results[0]))

for epoch_idx, epoch in enumerate(epochs_sweep):
    for alpha_idx, alpha in enumerate(sparsity_sweep):
        print(f"\n\nTraining with epoch={epoch}, alpha={alpha}")
        _, losses = train_autoencoder(
            dataset_dir=dataset_dir,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=epoch,
            R=R,
            alpha=alpha
        )

        print(f"Final loss: {losses['epoch_losses'][-1]:.6f}")
        print(f"Final reconstruction loss: {losses['reconstruction_losses'][-1]:.6f}")
        print(f"Final sparsity loss: {losses['sparsity_losses'][-1]:.6f}")

        results[epoch_idx][alpha_idx] = (
            losses['reconstruction_losses'][-1],
            sparsity_sweep[alpha_idx]
        )

plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(epochs_sweep)))

for epoch_idx, epoch in enumerate(epochs_sweep):
    sparsity_losses = []
    reconstruction_losses = []
    
    for alpha_idx, alpha in enumerate(sparsity_sweep):
        reconstruction_loss, sparsity_loss = results[epoch_idx][alpha_idx]
        reconstruction_losses.append(reconstruction_loss)
        sparsity_losses.append(sparsity_loss)
    
    plt.scatter(sparsity_losses, reconstruction_losses, 
               color=colors[epoch_idx], 
               label=f'{epoch} epochs',
               alpha=0.7, 
               s=60)
    
    plt.plot(sparsity_losses, reconstruction_losses, 
             color=colors[epoch_idx], 
             alpha=0.3, 
             linewidth=1)

plt.xlabel('Sparsity Weight (alpha)', fontsize=14)
plt.ylabel('Unexplained Variance', fontsize=14)
plt.title('Reconstruction-Sparsity Trade-off', fontsize=16)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.xscale('log')
plt.tight_layout()
plt.savefig('plot.png')
