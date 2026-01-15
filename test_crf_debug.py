#!/usr/bin/env python3
"""Quick test to debug CRF loss values"""

import torch
import sys
sys.path.insert(0, '/Users/ez/ecom_bert_crf/BERT-CRF-EVAL')

from models.layers.crf import CRF

# Test with small example
batch_size = 2
seq_len = 5
num_tags = 9

# Create CRF
crf = CRF(num_tags=num_tags, batch_first=True)

# Create random emissions (logits) - these should be small values initially
emissions = torch.randn(batch_size, seq_len, num_tags) * 0.1  # Small random values

# Create sample tags
tags = torch.randint(0, num_tags, (batch_size, seq_len))

# Create mask
mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

print("Emissions shape:", emissions.shape)
print("Emissions range:", emissions.min().item(), "to", emissions.max().item())
print("Tags shape:", tags.shape)
print("Tags:", tags)
print("Mask shape:", mask.shape)

# Compute loss
log_likelihood = crf(emissions, tags, mask)
loss = -log_likelihood

print("\nLog likelihood:", log_likelihood)
print("Loss (negated):", loss)
print("Loss mean:", loss.mean().item())

# Now test with larger emissions (like from untrained BERT)
print("\n" + "="*50)
print("Testing with larger emissions (like untrained model):")
emissions_large = torch.randn(batch_size, seq_len, num_tags) * 10  # Larger values
log_likelihood_large = crf(emissions_large, tags, mask)
loss_large = -log_likelihood_large
print("Log likelihood:", log_likelihood_large)
print("Loss (negated):", loss_large)
print("Loss mean:", loss_large.mean().item())
