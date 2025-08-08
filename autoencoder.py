import torch
from torch import nn
from torch.nn import functional as F
import os

torch.manual_seed(42)

class TiedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(TiedAutoencoder, self).__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, latent_dim))
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("relu"))

    def forward(self, x):
        z = F.linear(x, self.weights.T)
        z = F.relu(z)
        z = F.linear(z, self.weights)
        return z

    def get_latent_sum(self):
        return torch.abs(self.weights.sum())


dataset_dir = "./activations_chunks"
sample_file_path = os.path.join(dataset_dir, "layer_2_chunk_0.pt")
# assert os.path.exists(sample_file_path), "Dataset not found at {}".format(dataset_dir)
n_chunks = len(os.listdir(dataset_dir))

# config
batch_size = 512
learning_rate = 1e-3
num_epochs = 100
R = 100
alpha = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

first_chunk = torch.load(os.path.join(dataset_dir, "layer_2_chunk_0.pt")).to(
    dtype=torch.float32
)
input_dim = first_chunk.shape[-1]
latent_dim = input_dim * R

model = TiedAutoencoder(input_dim, latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

print(f"input_dim={input_dim}, latent_dim={latent_dim}")

for epoch in range(num_epochs):
    epoch_reconstruction_loss = 0.0
    epoch_sparsity_loss = 0.0
    epoch_loss = 0.0
    total_batches = 0

    for chunk_idx in range(n_chunks):
        dataset = torch.load(
            os.path.join(dataset_dir, f"layer_2_chunk_{chunk_idx}.pt")
        ).to(dtype=torch.float32)
        dataset = dataset.view(-1, dataset.shape[-1])

        # perform normalization
        dataset = (dataset - dataset.mean()) / dataset.std()
        
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=True
        )

        for batch in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()
            reconstructed = model(batch)

            reconstruction_loss = criterion(reconstructed, batch)
            sparsity_loss = alpha * model.get_latent_sum()

            loss = reconstruction_loss + sparsity_loss

            loss.backward()
            optimizer.step()

            epoch_reconstruction_loss += reconstruction_loss  # hm
            epoch_sparsity_loss += sparsity_loss
            epoch_loss += loss.item()
            total_batches += 1

    avg_epoch_loss = epoch_loss / total_batches
    avg_epoch_reconstruction_loss = epoch_reconstruction_loss / total_batches
    avg_epoch_sparsity_loss = epoch_sparsity_loss / total_batches

    print(
        f"Epoch {epoch+1}/{num_epochs} avg loss: {avg_epoch_loss:.6f}, avg recons. loss: {avg_epoch_reconstruction_loss:.6f}, avg sparsity loss: {avg_epoch_sparsity_loss:.6f}"
    )
