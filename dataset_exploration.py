"""
Dataset Exploration Script for Optimal Sequence Length Selection

This script analyzes the WikiText-2 dataset to help determine the optimal
sequence length for tokenization and model processing.
"""

import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformer_lens import HookedTransformer
from tqdm import tqdm
import pandas as pd
from collections import Counter
import torch
import sys
import os

# Add the current directory to path to import from script.py
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from script import streaming_sequence_generator  # Not actually used in this analysis

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load model and dataset
print("Loading model and dataset...")
model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device='cpu')
dataset = datasets.load_dataset("Salesforce/wikitext", name="wikitext-2-raw-v1", split="train[:30000]")

print(f"Dataset size: {len(dataset)} samples")
print(f"Model vocabulary size: {model.cfg.d_vocab}")
print(f"Model context length: {model.cfg.n_ctx}")

def analyze_text_lengths():
    """Analyze character lengths of text samples."""
    print("\n=== CHARACTER LENGTH ANALYSIS ===")
    
    text_lengths = []
    non_empty_texts = []
    
    for item in tqdm(dataset, desc="Analyzing text lengths"):
        text = item["text"].strip()
        if text:  # Skip empty texts
            text_lengths.append(len(text))
            non_empty_texts.append(text)
    
    text_lengths = np.array(text_lengths)
    
    print(f"Total samples: {len(dataset)}")
    print(f"Non-empty samples: {len(non_empty_texts)}")
    print(f"Empty samples: {len(dataset) - len(non_empty_texts)}")
    print(f"Character length stats:")
    print(f"  Mean: {np.mean(text_lengths):.1f}")
    print(f"  Median: {np.median(text_lengths):.1f}")
    print(f"  Std: {np.std(text_lengths):.1f}")
    print(f"  Min: {np.min(text_lengths)}")
    print(f"  Max: {np.max(text_lengths)}")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"Character length percentiles:")
    for p in percentiles:
        print(f"  {p}th: {np.percentile(text_lengths, p):.1f}")
    
    return text_lengths, non_empty_texts

def analyze_token_lengths(non_empty_texts):
    """Analyze token lengths of text samples."""
    print("\n=== TOKEN LENGTH ANALYSIS ===")
    
    token_lengths = []
    char_to_token_ratios = []
    
    for text in tqdm(non_empty_texts, desc="Tokenizing samples"):  # Use entire dataset
        tokens = model.tokenizer(text, add_special_tokens=False)['input_ids']
        token_count = len(tokens)
        char_count = len(text)
        
        token_lengths.append(token_count)
        if char_count > 0:
            char_to_token_ratios.append(char_count / token_count)
    
    token_lengths = np.array(token_lengths)
    char_to_token_ratios = np.array(char_to_token_ratios)
    
    print(f"Token length stats (from {len(token_lengths)} samples):")
    print(f"  Mean: {np.mean(token_lengths):.1f}")
    print(f"  Median: {np.median(token_lengths):.1f}")
    print(f"  Std: {np.std(token_lengths):.1f}")
    print(f"  Min: {np.min(token_lengths)}")
    print(f"  Max: {np.max(token_lengths)}")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"Token length percentiles:")
    for p in percentiles:
        print(f"  {p}th: {np.percentile(token_lengths, p):.1f}")
    
    print(f"\nCharacter-to-token ratio stats:")
    print(f"  Mean: {np.mean(char_to_token_ratios):.2f}")
    print(f"  Median: {np.median(char_to_token_ratios):.2f}")
    print(f"  This means ~{np.mean(char_to_token_ratios):.1f} characters per token on average")
    
    return token_lengths, char_to_token_ratios

def analyze_sequence_length_impact(token_lengths):
    """Analyze impact of different sequence lengths on data utilization."""
    print("\n=== SEQUENCE LENGTH IMPACT ANALYSIS ===")
    
    sequence_lengths = [10, 60, 110, 160, 210, 260, 310, 360, 410, 460, 510]
    
    results = []
    for seq_len in sequence_lengths:
        # How many samples would fit completely
        complete_fit = np.sum(token_lengths <= seq_len)
        complete_fit_pct = (complete_fit / len(token_lengths)) * 100
        
        # How many tokens would be lost due to truncation
        truncated_tokens = np.sum(np.maximum(0, token_lengths - seq_len))
        total_tokens = np.sum(token_lengths)
        token_loss_pct = (truncated_tokens / total_tokens) * 100
        
        # Average utilization (how much of each sequence is actual content vs padding)
        avg_utilization = np.mean(np.minimum(token_lengths, seq_len) / seq_len) * 100
        
        results.append({
            'seq_len': seq_len,
            'complete_fit': complete_fit,
            'complete_fit_pct': complete_fit_pct,
            'token_loss_pct': token_loss_pct,
            'avg_utilization': avg_utilization
        })
        
        print(f"Sequence length {seq_len}:")
        print(f"  Samples fitting completely: {complete_fit}/{len(token_lengths)} ({complete_fit_pct:.1f}%)")
        print(f"  Token loss from truncation: {token_loss_pct:.1f}%")
        print(f"  Average sequence utilization: {avg_utilization:.1f}%")
        print()
    
    return results

def analyze_samples_per_sequence(non_empty_texts):
    """Analyze how many original samples get combined into each sequence using streaming concatenation."""
    print("\n=== SAMPLES PER SEQUENCE ANALYSIS ===")
    
    sequence_lengths = [10, 60, 110, 160, 210, 260, 310, 360, 410, 460, 510]
    eos_token = model.tokenizer.eos_token or "<|endoftext|>"
    
    results = []
    
    for seq_len in tqdm(sequence_lengths, desc="Analyzing samples per sequence"):
        # Create a dataset-like object for analysis
        dataset_subset = [{"text": text} for text in non_empty_texts[:5000]]  # Use subset for speed
        
        # Track how many original samples contribute to each sequence
        token_buffer = []
        current_sample_count = 0
        sequence_sample_counts = []
        
        for item in dataset_subset:
            if not item["text"].strip():
                continue
                
            # Add this sample's tokens to buffer
            text_with_eos = eos_token + item["text"]
            tokens = model.tokenizer(text_with_eos, add_special_tokens=False)['input_ids']
            
            # Add tokens to buffer and increment sample count
            token_buffer.extend(tokens)
            current_sample_count += 1
            
            # Extract sequences while we have enough tokens (but don't save them)
            while len(token_buffer) >= seq_len:
                # Record how many samples contributed to this sequence
                sequence_sample_counts.append(current_sample_count)
                
                # Remove the sequence from buffer (but don't save it anywhere)
                token_buffer = token_buffer[seq_len:]
                
                # Reset sample count for next sequence
                current_sample_count = 0
        
        if sequence_sample_counts:
            median_samples = np.median(sequence_sample_counts)
            mean_samples = np.mean(sequence_sample_counts)
        else:
            median_samples = 0
            mean_samples = 0
        
        results.append({
            'seq_len': seq_len,
            'median_samples': median_samples,
            'mean_samples': mean_samples,
            'total_sequences': len(sequence_sample_counts)
        })
        
        print(f"Sequence length {seq_len}:")
        print(f"  Total sequences generated: {len(sequence_sample_counts)}")
        print(f"  Median samples per sequence: {median_samples:.1f}")
        print(f"  Mean samples per sequence: {mean_samples:.1f}")
        print()
    
    return results

def visualize_basic_distributions(text_lengths, token_lengths, char_to_token_ratios):
    """Create basic visualizations of the data distributions."""
    print("\n=== CREATING BASIC DISTRIBUTION VISUALIZATIONS ===")
    
    # Filter out zero-length values for the second row
    text_lengths_filtered = text_lengths[text_lengths > 0]
    token_lengths_filtered = token_lengths[token_lengths > 0]
    char_to_token_ratios_filtered = char_to_token_ratios[char_to_token_ratios > 0]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('WikiText-2 Dataset: Basic Distributions', fontsize=16, fontweight='bold')
    
    # ===== FIRST ROW: All Data (including zeros) =====
    
    # Character length distribution
    axes[0, 0].hist(text_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Character Length Distribution (All Data)')
    axes[0, 0].set_xlabel('Characters')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.median(text_lengths), color='red', linestyle='--', label=f'Median: {np.median(text_lengths):.0f}')
    axes[0, 0].legend()
    
    # Token length distribution
    axes[0, 1].hist(token_lengths, bins=50, alpha=0.7, color='salmon', edgecolor='black')
    axes[0, 1].set_title('Token Length Distribution (All Data)')
    axes[0, 1].set_xlabel('Tokens')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.median(token_lengths), color='red', linestyle='--', label=f'Median: {np.median(token_lengths):.0f}')
    axes[0, 1].legend()
    
    # Character-to-token ratio
    axes[0, 2].hist(char_to_token_ratios, bins=30, alpha=0.7, color='gold', edgecolor='black')
    axes[0, 2].set_title('Characters per Token Distribution (All Data)')
    axes[0, 2].set_xlabel('Characters per Token')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(np.median(char_to_token_ratios), color='red', linestyle='--', 
                      label=f'Median: {np.median(char_to_token_ratios):.2f}')
    axes[0, 2].legend()
    
    # ===== SECOND ROW: Filtered Data (excluding zeros) =====
    
    print(f"\n=== FILTERED DATA INFO ===")
    print(f"Character lengths (excluding zeros): {len(text_lengths_filtered)} samples")
    print(f"Token lengths (excluding zeros): {len(token_lengths_filtered)} samples") 
    print(f"Char-to-token ratios (excluding zeros): {len(char_to_token_ratios_filtered)} samples")
    
    # Character length distribution (filtered)
    axes[1, 0].hist(text_lengths_filtered, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    axes[1, 0].set_title('Character Length Distribution (Excluding Zeros)')
    axes[1, 0].set_xlabel('Characters')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].axvline(np.median(text_lengths_filtered), color='red', linestyle='--', 
                      label=f'Median: {np.median(text_lengths_filtered):.0f}')
    axes[1, 0].legend()
    
    # Token length distribution (filtered)
    axes[1, 1].hist(token_lengths_filtered, bins=50, alpha=0.7, color='lightsalmon', edgecolor='black')
    axes[1, 1].set_title('Token Length Distribution (Excluding Zeros)')
    axes[1, 1].set_xlabel('Tokens')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.median(token_lengths_filtered), color='red', linestyle='--', 
                      label=f'Median: {np.median(token_lengths_filtered):.0f}')
    axes[1, 1].legend()
    
    # Character-to-token ratio (filtered)
    axes[1, 2].hist(char_to_token_ratios_filtered, bins=30, alpha=0.7, color='lightyellow', edgecolor='black')
    axes[1, 2].set_title('Characters per Token Distribution (Excluding Zeros)')
    axes[1, 2].set_xlabel('Characters per Token')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].axvline(np.median(char_to_token_ratios_filtered), color='red', linestyle='--', 
                      label=f'Median: {np.median(char_to_token_ratios_filtered):.2f}')
    axes[1, 2].legend()
    
    # Print comparison statistics
    print(f"\n=== STATISTICS COMPARISON ===")
    print(f"Character lengths:")
    print(f"  All data - Mean: {np.mean(text_lengths):.1f}, Median: {np.median(text_lengths):.1f}")
    print(f"  Filtered - Mean: {np.mean(text_lengths_filtered):.1f}, Median: {np.median(text_lengths_filtered):.1f}")
    print(f"  Zero-length samples: {len(text_lengths) - len(text_lengths_filtered)}")
    
    print(f"\nToken lengths:")
    print(f"  All data - Mean: {np.mean(token_lengths):.1f}, Median: {np.median(token_lengths):.1f}")
    print(f"  Filtered - Mean: {np.mean(token_lengths_filtered):.1f}, Median: {np.median(token_lengths_filtered):.1f}")
    print(f"  Zero-length samples: {len(token_lengths) - len(token_lengths_filtered)}")
    
    print(f"\nChar-to-token ratios:")
    print(f"  All data - Mean: {np.mean(char_to_token_ratios):.2f}, Median: {np.median(char_to_token_ratios):.2f}")
    print(f"  Filtered - Mean: {np.mean(char_to_token_ratios_filtered):.2f}, Median: {np.median(char_to_token_ratios_filtered):.2f}")
    print(f"  Zero-ratio samples: {len(char_to_token_ratios) - len(char_to_token_ratios_filtered)}")
    
    plt.tight_layout()
    plt.savefig('dataset_analysis_basic.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_sequence_analysis(token_lengths, samples_per_seq_results):
    """Create sequence length analysis visualizations."""
    print("\n=== CREATING SEQUENCE ANALYSIS VISUALIZATIONS ===")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('WikiText-2 Dataset: Sequence Length Analysis', fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    seq_lengths = [result['seq_len'] for result in samples_per_seq_results]
    median_samples = [result['median_samples'] for result in samples_per_seq_results]
    mean_samples = [result['mean_samples'] for result in samples_per_seq_results]
    
    # Sequence length coverage
    coverage = [np.mean(token_lengths <= seq_len) * 100 for seq_len in seq_lengths]
    
    axes[0].plot(seq_lengths, coverage, 'o-', linewidth=2, markersize=8, color='purple')
    axes[0].set_title('Sample Coverage by Sequence Length')
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('% of Samples Fitting Completely')
    axes[0].grid(True, alpha=0.3)
    
    # Add annotations for key points
    for i, (seq_len, cov) in enumerate(zip(seq_lengths, coverage)):
        if i % 2 == 0:  # Annotate every other point to avoid clutter
            axes[0].annotate(f'{cov:.1f}%', (seq_len, cov), 
                           textcoords="offset points", xytext=(0,10), ha='center')
    
    # Samples per sequence analysis
    axes[1].plot(seq_lengths, median_samples, 'o-', linewidth=2, markersize=8, 
                color='darkgreen', label='Median')
    axes[1].plot(seq_lengths, mean_samples, 's-', linewidth=2, markersize=8, 
                color='darkred', label='Mean')
    axes[1].set_title('Original Samples per Generated Sequence')
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Number of Original Samples')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Token utilization efficiency
    utilization = []
    for seq_len in seq_lengths:
        avg_util = np.mean(np.minimum(token_lengths, seq_len) / seq_len) * 100
        utilization.append(avg_util)
    
    axes[2].plot(seq_lengths, utilization, 'o-', linewidth=2, markersize=8, color='orange')
    axes[2].set_title('Average Token Utilization')
    axes[2].set_xlabel('Sequence Length')
    axes[2].set_ylabel('Utilization %')
    axes[2].grid(True, alpha=0.3)
    
    # Add annotations for utilization
    for i, (seq_len, util) in enumerate(zip(seq_lengths, utilization)):
        if i % 2 == 0:  # Annotate every other point
            axes[2].annotate(f'{util:.1f}%', (seq_len, util), 
                           textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis_sequences.png', dpi=300, bbox_inches='tight')
    plt.show()

def sample_texts_analysis(non_empty_texts):
    """Show some sample texts to understand content structure."""
    print("\n=== SAMPLE TEXTS ANALYSIS ===")
    
    # Show some samples of different lengths
    lengths = [len(model.tokenizer(text, add_special_tokens=False)['input_ids']) for text in non_empty_texts[:1000]]
    
    # Short samples
    short_indices = [i for i, length in enumerate(lengths) if length < 50]
    medium_indices = [i for i, length in enumerate(lengths) if 100 <= length <= 200]
    long_indices = [i for i, length in enumerate(lengths) if length > 500]
    
    print("SHORT SAMPLE (< 50 tokens):")
    if short_indices:
        idx = short_indices[0]
        text = non_empty_texts[idx]
        tokens = model.tokenizer(text, add_special_tokens=False)['input_ids']
        print(f"Length: {len(tokens)} tokens, {len(text)} characters")
        print(f"Text: {repr(text[:200])}")
        print()
    
    print("MEDIUM SAMPLE (100-200 tokens):")
    if medium_indices:
        idx = medium_indices[0]
        text = non_empty_texts[idx]
        tokens = model.tokenizer(text, add_special_tokens=False)['input_ids']
        print(f"Length: {len(tokens)} tokens, {len(text)} characters")
        print(f"Text: {repr(text[:300])}")
        print()
    
    print("LONG SAMPLE (> 500 tokens):")
    if long_indices:
        idx = long_indices[0]
        text = non_empty_texts[idx]
        tokens = model.tokenizer(text, add_special_tokens=False)['input_ids']
        print(f"Length: {len(tokens)} tokens, {len(text)} characters")
        print(f"Text: {repr(text[:400])}")
        print()

def recommend_sequence_length(results, token_lengths):
    """Provide recommendations for sequence length selection."""
    print("\n=== SEQUENCE LENGTH RECOMMENDATIONS ===")
    
    print("Considerations for sequence length selection:")
    print("1. Model context window:", model.cfg.n_ctx)
    print("2. Memory constraints: Longer sequences = more GPU memory")
    print("3. Data efficiency: Balance between utilization and loss")
    print("4. Training stability: Consistent sequence lengths help")
    print()
    
    # Find optimal based on different criteria
    df = pd.DataFrame(results)
    
    # Best for data preservation (minimal token loss)
    best_preservation = df.loc[df['token_loss_pct'].idxmin()]
    
    # Best balance (high utilization, reasonable fit)
    df['balance_score'] = df['avg_utilization'] * df['complete_fit_pct'] / 100
    best_balance = df.loc[df['balance_score'].idxmax()]
    
    # Most common sequence length that fits most data
    popular_choice = df[df['complete_fit_pct'] >= 80].iloc[0] if len(df[df['complete_fit_pct'] >= 80]) > 0 else df.iloc[2]
    
    print("RECOMMENDATIONS:")
    print(f"• For maximum data preservation: {best_preservation['seq_len']} tokens")
    print(f"  - Loses only {best_preservation['token_loss_pct']:.1f}% of tokens")
    print(f"  - {best_preservation['complete_fit_pct']:.1f}% of samples fit completely")
    print()
    
    print(f"• For balanced efficiency: {best_balance['seq_len']} tokens")
    print(f"  - {best_balance['avg_utilization']:.1f}% average utilization")
    print(f"  - {best_balance['complete_fit_pct']:.1f}% of samples fit completely")
    print(f"  - {best_balance['token_loss_pct']:.1f}% token loss")
    print()
    
    print(f"• Popular choice (80%+ coverage): {popular_choice['seq_len']} tokens")
    print(f"  - {popular_choice['complete_fit_pct']:.1f}% of samples fit completely")
    print(f"  - {popular_choice['token_loss_pct']:.1f}% token loss")
    print()
    
    # Memory considerations
    median_tokens = np.median(token_lengths)
    print("MEMORY CONSIDERATIONS:")
    print(f"• Dataset median token length: {median_tokens:.0f}")
    print(f"• For batch_size=8 with seq_len=256: ~{8 * 256 * 4 / 1024:.1f}KB per batch (fp32)")
    print(f"• Model context limit: {model.cfg.n_ctx} tokens")
    print()
    
    print("FINAL RECOMMENDATION:")
    if median_tokens <= 256:
        print("• Use 256 tokens - good balance for this dataset")
    elif median_tokens <= 512:
        print("• Use 512 tokens - captures most content without excessive padding")
    else:
        print("• Use 1024 tokens - necessary for longer content, watch memory usage")

def main():
    """Run the complete dataset analysis."""
    print("Starting comprehensive dataset analysis...")
    print("=" * 60)
    
    # Analyze text lengths
    text_lengths, non_empty_texts = analyze_text_lengths()
    
    # Analyze token lengths
    token_lengths, char_to_token_ratios = analyze_token_lengths(non_empty_texts)
    
    # Analyze sequence length impact
    results = analyze_sequence_length_impact(token_lengths)
    
    # Analyze samples per sequence
    samples_per_seq_results = analyze_samples_per_sequence(non_empty_texts)
    
    # Show sample texts
    sample_texts_analysis(non_empty_texts)
    
    # Create visualizations (two separate images)
    visualize_basic_distributions(text_lengths, token_lengths, char_to_token_ratios)
    visualize_sequence_analysis(token_lengths, samples_per_seq_results)
    
    # Provide recommendations
    recommend_sequence_length(results, token_lengths)
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check 'dataset_analysis_basic.png' and 'dataset_analysis_sequences.png' for visualizations.")

if __name__ == "__main__":
    main()
