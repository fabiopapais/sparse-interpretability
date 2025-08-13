"""
Sparse Autoencoder Implementation with L0/L1 Regularization
Organized version with train/test/validation splits and model saving
"""

import torch
from torch import nn
from torch.nn import functional as F
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import json
from datetime import datetime

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ========================================
# CONFIGURATION
# ========================================

class Config:
    """Configuration class for hyperparameters and settings"""
    
    # Dataset
    dataset_dir = "./activations_chunks"
    
    # Training parameters
    batch_size = 4
    learning_rate = 1e-3
    num_epochs = 1
    R = 10  # Hidden layer expansion ratio
    alpha = 5e-1  # Sparsity penalty weight
    
    # Data splits
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    # Regularization mode (0=L0, 1=L1)
    loss_mode = 1
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Debug
    DEBUG = True
    
    # Model saving
    save_dir = "./saved_models"
    save_best_model = True

# ========================================
# UTILITY FUNCTIONS
# ========================================

def debug_print(msg):
    """Print debug messages if DEBUG flag is enabled"""
    if Config.DEBUG:
        print(f"[DEBUG] {msg}")

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        debug_print(f"Created directory: {directory}")

def create_timestamped_dir(base_dir, prefix=""):
    """Create a timestamped directory to avoid overwrites"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        dir_name = f"{prefix}_{timestamp}"
    else:
        dir_name = timestamp
    
    full_path = os.path.join(base_dir, dir_name)
    ensure_dir(full_path)
    return full_path

# ========================================
# REGULARIZATION CLASSES
# ========================================

class L0Regularizer(nn.Module):
    """L0 regularization using concrete random variables"""
    
    def __init__(self, size, droprate_init=0.5, beta=2./3., gamma=-0.1, zeta=1.1):
        super().__init__()
        init_val = -torch.log(torch.tensor(1.0 / droprate_init - 1.0))
        self.qz_loga = nn.Parameter(torch.ones(size) * init_val)
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta

    def _sample_z(self, training=True):
        if training:
            u = torch.rand_like(self.qz_loga)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.qz_loga) / self.beta)
        else:
            s = torch.sigmoid(self.qz_loga)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        return torch.clamp(s_bar, 0, 1)

    def expected_l0(self):
        """Calculate expected number of active units"""
        gamma_tensor = torch.tensor(self.gamma, device=self.qz_loga.device)
        zeta_tensor = torch.tensor(self.zeta, device=self.qz_loga.device)
        beta_tensor = torch.tensor(self.beta, device=self.qz_loga.device)
        
        # Ensure we don't take log of negative numbers
        ratio = -gamma_tensor / zeta_tensor
        if ratio <= 0:
            s = torch.sigmoid(self.qz_loga)
        else:
            s = torch.sigmoid(self.qz_loga - beta_tensor * torch.log(ratio))
        return torch.sum(s)

    def forward(self, c, training=True):
        z = self._sample_z(training)
        z = z.to(c.device)
        return c * z

# ========================================
# LOSS FUNCTIONS
# ========================================

def sparsity_term(c: torch.Tensor, mode: int = 0, l0_gates=None) -> torch.Tensor:
    """Calculate the sparsity term based on the regularization mode"""
    if mode == 0:
        if l0_gates is None:
            raise ValueError("L0 regularizer required for mode=0")
        # For L0, return expected L0 count (gates are already applied in forward pass)
        return l0_gates.expected_l0()
    elif mode == 1:
        return torch.sum(torch.abs(c))  # L1 norm
    else:
        raise ValueError("Invalid mode. Use 0 for L0 regularization or 1 for L1 regularization.")

def custom_loss(x, x_hat, c_original, c_used_for_decoding, alpha, mode=0, l0_gates=None):
    """Calculate total loss with reconstruction and sparsity terms"""
    # Reconstruction loss
    reconstruction_loss = F.mse_loss(x_hat, x, reduction='sum')
    
    # Sparsity penalty - use appropriate activations based on mode
    if mode == 0:
        # For L0, use the sparsity term on the gated activations
        sparsity_loss = sparsity_term(c_used_for_decoding, mode, l0_gates)
    else:
        # For L1, use the original activations (before any gating)
        sparsity_loss = sparsity_term(c_original, mode, l0_gates)
    
    total_loss = reconstruction_loss + alpha * sparsity_loss
    return total_loss

# ========================================
# MODEL DEFINITION
# ========================================

class TiedAutoencoder(nn.Module):
    """Tied-weight autoencoder with row-wise weight normalization"""
    
    def __init__(self, input_dim, latent_dim, l0_gates=None):
        super(TiedAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.l0_gates = l0_gates  # Store L0 regularizer for use in forward pass
        
        # Weight matrix: dhid x din
        self.weights = nn.Parameter(torch.randn(latent_dim, input_dim))
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(latent_dim))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.weights, gain=nn.init.calculate_gain("relu"))
        
    def normalize_weights(self):
        """Normalize weights row-wise as specified in the paper"""
        with torch.no_grad():
            self.weights.data = F.normalize(self.weights.data, p=2, dim=1)

    def forward(self, x, apply_l0_gates=True):
        """Forward pass: encode and decode"""
        # Ensure weights are normalized row-wise
        normalized_weights = F.normalize(self.weights, p=2, dim=1)
        
        # Encode: c = ReLU(Mx + b)
        c = F.linear(x, normalized_weights, self.bias)
        c = F.relu(c)
        
        # Apply L0 gates if available and requested
        if self.l0_gates is not None and apply_l0_gates:
            c_gated = self.l0_gates(c, training=self.training)
            # Use gated activations for decoding
            x_hat = F.linear(c_gated, normalized_weights.T)
            return x_hat, c, c_gated  # Return original and gated activations
        else:
            # Decode with original activations: x̂ = M^T c  
            x_hat = F.linear(c, normalized_weights.T)
            return x_hat, c, c  # Return original activations twice for consistency

    def get_latent_sum(self):
        """Get sum of absolute weights for analysis"""
        return torch.abs(self.weights.sum())

# ========================================
# DATA LOADING AND SPLITTING
# ========================================

class DataManager:
    """Manage dataset loading and train/val/test splits"""
    
    def __init__(self, config):
        self.config = config
        self.chunk_files = []
        self.train_chunks = []
        self.val_chunks = []
        self.test_chunks = []
        
    def load_chunk_files(self):
        """Load list of available chunk files"""
        debug_print("Loading dataset directory...")
        self.chunk_files = [f for f in os.listdir(self.config.dataset_dir) 
                           if f.startswith("layer_2_chunk_") and f.endswith(".pt")]
        self.chunk_files.sort()  # Ensure consistent ordering
        debug_print(f"Found {len(self.chunk_files)} chunks in dataset")
        return len(self.chunk_files)
    
    def split_chunks(self):
        """Split chunks into train/validation/test sets"""
        n_chunks = len(self.chunk_files)
        
        # Create indices for splitting
        indices = list(range(n_chunks))
        
        # First split: train vs (val + test)
        train_indices, temp_indices = train_test_split(
            indices, 
            test_size=(self.config.val_ratio + self.config.test_ratio),
            random_state=42
        )
        
        # Second split: val vs test
        val_indices, test_indices = train_test_split(
            temp_indices,
            test_size=self.config.test_ratio / (self.config.val_ratio + self.config.test_ratio),
            random_state=42
        )
        
        # Get actual chunk names
        self.train_chunks = [self.chunk_files[i] for i in train_indices]
        self.val_chunks = [self.chunk_files[i] for i in val_indices]
        self.test_chunks = [self.chunk_files[i] for i in test_indices]
        
        debug_print(f"Data split - Train: {len(self.train_chunks)}, Val: {len(self.val_chunks)}, Test: {len(self.test_chunks)}")
        
    def get_data_splits(self):
        """Get train/val/test chunk lists"""
        return self.train_chunks, self.val_chunks, self.test_chunks
    
    def load_chunk(self, chunk_name):
        """Load and preprocess a single chunk"""
        chunk_path = os.path.join(self.config.dataset_dir, chunk_name)
        dataset = torch.load(chunk_path).to(dtype=torch.float32)
        dataset = dataset.view(-1, dataset.shape[-1])
        
        # Normalize
        dataset = (dataset - dataset.mean()) / dataset.std()
        
        return dataset

# ========================================
# TRAINING AND VALIDATION
# ========================================

class Trainer:
    """Training manager with validation and model saving"""
    
    def __init__(self, model, l0_gates, optimizer, config, data_manager):
        self.model = model
        self.l0_gates = l0_gates
        self.optimizer = optimizer
        self.config = config
        self.data_manager = data_manager
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.chunk_losses = []
        self.chunk_reconstruction_losses = []
        self.chunk_sparsity_losses = []
        self.chunk_l1_losses = []  # Always track L1 for comparison
        self.chunk_l0_losses = []  # Always track L0 for comparison
        self.chunk_identifiers = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Create timestamped save directory
        base_save_dir = self.config.save_dir
        regularization_type = "L0" if self.config.loss_mode == 0 else "L1"
        self.timestamped_save_dir = create_timestamped_dir(base_save_dir, f"autoencoder_{regularization_type}")
        debug_print(f"Created timestamped save directory: {self.timestamped_save_dir}")
    
    def train_epoch(self, epoch, chunk_list):
        """Train for one epoch"""
        self.model.train()
        if self.l0_gates:
            self.l0_gates.train()
            
        epoch_start_time = time.time()
        epoch_loss = 0.0
        epoch_reconstruction_loss = 0.0
        epoch_sparsity_loss = 0.0
        total_batches = 0
        
        debug_print(f"Starting epoch {epoch+1}/{self.config.num_epochs}")
        
        for chunk_idx, chunk_name in enumerate(tqdm(chunk_list, desc=f"Training Chunks (Epoch {epoch+1})", leave=False)):
            chunk_start_time = time.time()
            debug_print(f"Loading chunk {chunk_name}")
            
            # Load and prepare data
            dataset = self.data_manager.load_chunk(chunk_name)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=True
            )
            
            # Track chunk-level losses
            chunk_loss = 0.0
            chunk_recon_loss = 0.0
            chunk_sparse_loss = 0.0
            chunk_l1_loss = 0.0
            chunk_l0_loss = 0.0
            chunk_batches = 0
            
            debug_print(f"Processing {len(dataloader)} batches for chunk {chunk_name}")
            
            for batch in dataloader:
                batch = batch.to(self.config.device)
                
                self.optimizer.zero_grad()
                reconstructed, latent_codes_original, latent_codes_used = self.model(batch)
                
                # Calculate loss
                loss = custom_loss(
                    batch, reconstructed, latent_codes_original, latent_codes_used,
                    self.config.alpha, self.config.loss_mode, self.l0_gates
                )
                
                loss.backward()
                self.optimizer.step()
                
                # Calculate components for logging
                with torch.no_grad():
                    recon_loss = F.mse_loss(reconstructed, batch, reduction='sum')
                    sparse_loss = sparsity_term(
                        latent_codes_original if self.config.loss_mode == 1 else latent_codes_used,
                        self.config.loss_mode, self.l0_gates
                    )
                    
                    # Always calculate both L1 and L0 for comparison plotting
                    l1_loss = torch.sum(torch.abs(latent_codes_original))  # L1 on original activations
                    if self.l0_gates is not None:
                        l0_loss = self.l0_gates.expected_l0()  # L0 expected count
                    else:
                        l0_loss = torch.tensor(0.0)  # Fallback if no L0 gates
                
                # Accumulate losses
                epoch_loss += loss.item()
                epoch_reconstruction_loss += recon_loss.item()
                epoch_sparsity_loss += (self.config.alpha * sparse_loss).item()
                total_batches += 1
                
                chunk_loss += loss.item()
                chunk_recon_loss += recon_loss.item()
                chunk_sparse_loss += (self.config.alpha * sparse_loss).item()
                chunk_l1_loss += (self.config.alpha * l1_loss).item()
                chunk_l0_loss += (self.config.alpha * l0_loss).item()
                chunk_batches += 1
            
            # Store chunk losses
            avg_chunk_loss = chunk_loss / chunk_batches
            avg_chunk_recon = chunk_recon_loss / chunk_batches
            avg_chunk_sparse = chunk_sparse_loss / chunk_batches
            avg_chunk_l1 = chunk_l1_loss / chunk_batches
            avg_chunk_l0 = chunk_l0_loss / chunk_batches
            
            self.chunk_losses.append(avg_chunk_loss)
            self.chunk_reconstruction_losses.append(avg_chunk_recon)
            self.chunk_sparsity_losses.append(avg_chunk_sparse)
            self.chunk_l1_losses.append(avg_chunk_l1)
            self.chunk_l0_losses.append(avg_chunk_l0)
            self.chunk_identifiers.append((epoch + 1, chunk_idx, chunk_name))
            
            chunk_time = time.time() - chunk_start_time
            print(f"  Chunk {chunk_name} - Loss: {avg_chunk_loss:.6f}, Recons: {avg_chunk_recon:.6f}, Sparsity: {avg_chunk_sparse:.6f} ({chunk_time:.2f}s)")
        
        # Calculate epoch averages
        avg_epoch_loss = epoch_loss / total_batches
        avg_epoch_recon = epoch_reconstruction_loss / total_batches
        avg_epoch_sparse = epoch_sparsity_loss / total_batches
        
        self.train_losses.append(avg_epoch_loss)
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{self.config.num_epochs} - Train Loss: {avg_epoch_loss:.6f}, Recons: {avg_epoch_recon:.6f}, Sparsity: {avg_epoch_sparse:.6f} ({epoch_time:.2f}s)")
        
        return avg_epoch_loss
    
    def validate(self, epoch, chunk_list):
        """Validate the model"""
        self.model.eval()
        if self.l0_gates:
            self.l0_gates.eval()
            
        val_loss = 0.0
        val_reconstruction_loss = 0.0
        val_sparsity_loss = 0.0
        total_batches = 0
        
        debug_print(f"Starting validation for epoch {epoch+1}")
        
        with torch.no_grad():
            for chunk_name in tqdm(chunk_list, desc=f"Validation Chunks (Epoch {epoch+1})", leave=False):
                dataset = self.data_manager.load_chunk(chunk_name)
                dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=True
                )
                
                for batch in dataloader:
                    batch = batch.to(self.config.device)
                    
                    reconstructed, latent_codes_original, latent_codes_used = self.model(batch)
                    
                    # Calculate loss
                    loss = custom_loss(
                        batch, reconstructed, latent_codes_original, latent_codes_used,
                        self.config.alpha, self.config.loss_mode, self.l0_gates
                    )
                    
                    # Calculate components
                    recon_loss = F.mse_loss(reconstructed, batch, reduction='sum')
                    sparse_loss = sparsity_term(
                        latent_codes_original if self.config.loss_mode == 1 else latent_codes_used,
                        self.config.loss_mode, self.l0_gates
                    )
                    
                    val_loss += loss.item()
                    val_reconstruction_loss += recon_loss.item()
                    val_sparsity_loss += (self.config.alpha * sparse_loss).item()
                    total_batches += 1
        
        avg_val_loss = val_loss / total_batches
        avg_val_recon = val_reconstruction_loss / total_batches
        avg_val_sparse = val_sparsity_loss / total_batches
        
        self.val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{self.config.num_epochs} - Val Loss: {avg_val_loss:.6f}, Recons: {avg_val_recon:.6f}, Sparsity: {avg_val_sparse:.6f}")
        
        # Save best model
        if self.config.save_best_model and avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.best_epoch = epoch + 1
            self.save_model(epoch, is_best=True)
            print(f"  New best validation loss: {self.best_val_loss:.6f}")
        
        return avg_val_loss
    
    def save_model(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': {
                'input_dim': self.model.input_dim,
                'latent_dim': self.model.latent_dim,
                'alpha': self.config.alpha,
                'loss_mode': self.config.loss_mode,
                'R': self.config.R
            },
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        
        if self.l0_gates:
            checkpoint['l0_gates_state_dict'] = self.l0_gates.state_dict()
        
        # Save current model
        model_path = os.path.join(self.timestamped_save_dir, f'autoencoder_epoch_{epoch+1}.pt')
        torch.save(checkpoint, model_path)
        debug_print(f"Saved model checkpoint: {model_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.timestamped_save_dir, 'autoencoder_best.pt')
            torch.save(checkpoint, best_path)
            debug_print(f"Saved best model: {best_path}")

# ========================================
# VISUALIZATION AND ANALYSIS
# ========================================

class Visualizer:
    """Handle plotting and visualization"""
    
    def __init__(self, config):
        self.config = config
    
    def plot_losses(self, trainer):
        """Plot loss curves per chunk and epoch"""
        debug_print("Creating loss plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Chunk-level plots
        x_values = list(range(len(trainer.chunk_losses)))
        
        # Plot 1: Total Loss per Chunk
        axes[0, 0].plot(x_values, trainer.chunk_losses, 'b-', linewidth=1, alpha=0.7)
        axes[0, 0].set_title('Total Loss per Chunk')
        axes[0, 0].set_xlabel('Chunk Index')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Reconstruction Loss per Chunk
        axes[0, 1].plot(x_values, trainer.chunk_reconstruction_losses, 'r-', linewidth=1, alpha=0.7)
        axes[0, 1].set_title('Reconstruction Loss per Chunk')
        axes[0, 1].set_xlabel('Chunk Index')
        axes[0, 1].set_ylabel('Reconstruction Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: L1 vs L0 Sparsity Comparison
        current_mode = "L0" if self.config.loss_mode == 0 else "L1"
        axes[0, 2].plot(x_values, trainer.chunk_l1_losses, 'g-', linewidth=2, alpha=0.8, label='L1 Sparsity')
        axes[0, 2].plot(x_values, trainer.chunk_l0_losses, 'm-', linewidth=2, alpha=0.8, label='L0 Sparsity')
        axes[0, 2].plot(x_values, trainer.chunk_sparsity_losses, 'k--', linewidth=2, alpha=0.9, 
                       label=f'Used ({current_mode})')
        axes[0, 2].set_title('L1 vs L0 Sparsity Comparison')
        axes[0, 2].set_xlabel('Chunk Index')
        axes[0, 2].set_ylabel('Sparsity Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Epoch-level plots
        if len(trainer.train_losses) > 0:
            epochs = list(range(1, len(trainer.train_losses) + 1))
            
            # Plot 4: Training vs Validation Loss
            axes[1, 0].plot(epochs, trainer.train_losses, 'b-', label='Training', linewidth=2)
            if len(trainer.val_losses) > 0:
                axes[1, 0].plot(epochs, trainer.val_losses, 'r-', label='Validation', linewidth=2)
            axes[1, 0].set_title('Training vs Validation Loss')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 5: Normalized losses comparison
            if len(trainer.chunk_losses) > 0:
                norm_total = np.array(trainer.chunk_losses) / np.max(trainer.chunk_losses)
                norm_recons = np.array(trainer.chunk_reconstruction_losses) / np.max(trainer.chunk_reconstruction_losses)
                
                # Normalize L1 and L0 for comparison
                max_l1 = np.max(trainer.chunk_l1_losses) if np.max(trainer.chunk_l1_losses) > 0 else 1
                max_l0 = np.max(trainer.chunk_l0_losses) if np.max(trainer.chunk_l0_losses) > 0 else 1
                norm_l1 = np.array(trainer.chunk_l1_losses) / max_l1
                norm_l0 = np.array(trainer.chunk_l0_losses) / max_l0
                
                axes[1, 1].plot(x_values, norm_total, 'b-', label='Total (normalized)', alpha=0.7)
                axes[1, 1].plot(x_values, norm_recons, 'r-', label='Reconstruction (normalized)', alpha=0.7)
                axes[1, 1].plot(x_values, norm_l1, 'g-', label='L1 (normalized)', alpha=0.7)
                axes[1, 1].plot(x_values, norm_l0, 'm-', label='L0 (normalized)', alpha=0.7)
                axes[1, 1].set_title('Normalized Losses Comparison')
                axes[1, 1].set_xlabel('Chunk Index')
                axes[1, 1].set_ylabel('Normalized Loss')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            # Plot 6: Sparsity ratio (L0/L1) analysis
            if len(trainer.chunk_l1_losses) > 0 and len(trainer.chunk_l0_losses) > 0:
                # Calculate ratio where L1 > 0 to avoid division by zero
                l1_array = np.array(trainer.chunk_l1_losses)
                l0_array = np.array(trainer.chunk_l0_losses)
                ratio = np.where(l1_array > 1e-8, l0_array / l1_array, 0)
                
                axes[1, 2].plot(x_values, ratio, 'purple', linewidth=1, alpha=0.7)
                axes[1, 2].set_title('L0/L1 Sparsity Ratio per Chunk')
                axes[1, 2].set_xlabel('Chunk Index')
                axes[1, 2].set_ylabel('L0/L1 Ratio')
                axes[1, 2].grid(True, alpha=0.3)
            else:
                # Fallback: show loss distribution
                axes[1, 2].hist(trainer.chunk_losses, bins=20, alpha=0.7, edgecolor='black')
                axes[1, 2].set_title('Distribution of Chunk Losses')
                axes[1, 2].set_xlabel('Loss')
                axes[1, 2].set_ylabel('Frequency')
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(trainer.timestamped_save_dir, f'loss_curves_alpha_{self.config.alpha}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        debug_print(f"Loss plots saved as: {plot_path}")
    
    def print_summary(self, trainer, model):
        """Print training summary and statistics"""
        debug_print("Printing training summary...")
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"Configuration:")
        print(f"  - Epochs: {self.config.num_epochs}")
        print(f"  - Batch size: {self.config.batch_size}")
        print(f"  - Learning rate: {self.config.learning_rate}")
        print(f"  - Alpha (sparsity): {self.config.alpha}")
        print(f"  - Loss mode: {'L0' if self.config.loss_mode == 0 else 'L1'}")
        print(f"  - Model dimensions: {model.input_dim} → {model.latent_dim}")
        print(f"  - Device: {self.config.device}")
        
        if len(trainer.chunk_losses) > 0:
            print(f"\nChunk-level Statistics:")
            print(f"  - Total chunks processed: {len(trainer.chunk_losses)}")
            print(f"  - Avg loss: {np.mean(trainer.chunk_losses):.6f} ± {np.std(trainer.chunk_losses):.6f}")
            print(f"  - Min loss: {min(trainer.chunk_losses):.6f}")
            print(f"  - Max loss: {max(trainer.chunk_losses):.6f}")
        
        if len(trainer.train_losses) > 0:
            print(f"\nEpoch-level Statistics:")
            print(f"  - Final training loss: {trainer.train_losses[-1]:.6f}")
            if len(trainer.val_losses) > 0:
                print(f"  - Final validation loss: {trainer.val_losses[-1]:.6f}")
                print(f"  - Best validation loss: {trainer.best_val_loss:.6f} (epoch {trainer.best_epoch})")
        
        print(f"\nModel Parameters:")
        print(f"  - Bias values (first 10): {model.bias.data[:10].cpu().numpy()}")
        print("="*80)

# ========================================
# MAIN TRAINING FUNCTION
# ========================================

def main():
    """Main training function"""
    config = Config()
    
    debug_print(f"Using device: {config.device}")
    debug_print(f"Regularization mode: {'L0' if config.loss_mode == 0 else 'L1'}")
    
    # Initialize data manager
    data_manager = DataManager(config)
    n_chunks = data_manager.load_chunk_files()
    data_manager.split_chunks()
    
    # Get data splits
    train_chunks, val_chunks, test_chunks = data_manager.get_data_splits()
    
    # Get dimensions from first chunk
    debug_print("Loading first chunk to get dimensions...")
    first_chunk = data_manager.load_chunk(train_chunks[0])
    input_dim = first_chunk.shape[-1]
    latent_dim = input_dim * config.R
    
    debug_print(f"Model dimensions: input_dim={input_dim}, latent_dim={latent_dim}")
    
    # Initialize L0 regularizer if needed
    l0_gates = None
    if config.loss_mode == 0:
        l0_gates = L0Regularizer(size=latent_dim).to(config.device)
        debug_print("L0 regularizer created and moved to device")
    
    # Initialize model (pass L0 gates to model)
    model = TiedAutoencoder(input_dim, latent_dim, l0_gates).to(config.device)
    
    # Setup optimizer
    if config.loss_mode == 0 and l0_gates is not None:
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(l0_gates.parameters()), 
            lr=config.learning_rate
        )
        debug_print("Optimizer includes L0 regularizer parameters")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        debug_print("Optimizer includes only model parameters")
    
    print(f"Model created: input_dim={input_dim}, latent_dim={latent_dim}")
    print(f"Using {'L0' if config.loss_mode == 0 else 'L1'} regularization")
    
    # Initialize trainer
    trainer = Trainer(model, l0_gates, optimizer, config, data_manager)
    
    # Training loop
    debug_print("Starting training loop...")
    for epoch in range(config.num_epochs):
        # Train
        train_loss = trainer.train_epoch(epoch, train_chunks)
        
        # Validate
        val_loss = trainer.validate(epoch, val_chunks)
        
        # Save model checkpoint
        trainer.save_model(epoch)
    
    debug_print("Training completed!")
    
    # Visualization and analysis
    visualizer = Visualizer(config)
    visualizer.plot_losses(trainer)
    visualizer.print_summary(trainer, model)
    
    # Save final training log
    log_path = os.path.join(trainer.timestamped_save_dir, f'training_log_{"L0" if config.loss_mode == 0 else "L1"}.json')
    training_log = {
        'config': {
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'num_epochs': config.num_epochs,
            'R': config.R,
            'alpha': config.alpha,
            'loss_mode': config.loss_mode,
            'input_dim': input_dim,
            'latent_dim': latent_dim
        },
        'data_splits': {
            'train_chunks': train_chunks,
            'val_chunks': val_chunks,
            'test_chunks': test_chunks
        },
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'best_val_loss': trainer.best_val_loss,
        'best_epoch': trainer.best_epoch,
        'chunk_losses': trainer.chunk_losses,
        'chunk_reconstruction_losses': trainer.chunk_reconstruction_losses,
        'chunk_sparsity_losses': trainer.chunk_sparsity_losses,
        'chunk_l1_losses': trainer.chunk_l1_losses,
        'chunk_l0_losses': trainer.chunk_l0_losses,
        'save_directory': trainer.timestamped_save_dir
    }
    
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    debug_print(f"Training log saved: {log_path}")
    
    return trainer.timestamped_save_dir  # Return save directory for reference

def run_both_modes():
    """Run training with both L0 and L1 regularization for comparison"""
    print("="*80)
    print("RUNNING COMPARISON: L0 vs L1 REGULARIZATION")
    print("="*80)
    
    # Save original config
    original_mode = Config.loss_mode
    saved_directories = []
    
    for mode, name in [(0, "L0"), (1, "L1")]:
        print(f"\n{'='*40}")
        print(f"TRAINING WITH {name} REGULARIZATION")
        print(f"{'='*40}")
        
        # Update config
        Config.loss_mode = mode
        
        # Run training and collect save directory
        save_dir = main()
        saved_directories.append((name, save_dir))
        
        print(f"{name} regularization training completed!")
        print(f"Results saved in: {save_dir}")
    
    # Restore original config
    Config.loss_mode = original_mode
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("Results saved in the following directories:")
    for name, directory in saved_directories:
        print(f"  {name}: {directory}")
    print("Check the saved models and plots to compare L0 vs L1 regularization")
    print("="*80)

if __name__ == "__main__":
    # To run single mode, use:
    main()
    
    # To compare both L0 and L1 regularization, uncomment:
    # run_both_modes()
