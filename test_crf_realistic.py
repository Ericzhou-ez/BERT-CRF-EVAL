#!/usr/bin/env python3
"""Minimal test to reproduce the issue"""

import torch
import sys
sys.path.insert(0, '/Users/ez/ecom_bert_crf/BERT-CRF-EVAL')

from models.layers.crf import CRF

# Simulate what happens in training
batch_size = 16
seq_len = 128
num_tags = 9

# Create CRF with batch_first=True (like in the model)
crf = CRF(num_tags=num_tags, batch_first=True)

# Create random emissions (like from BERT classifier)
# BERT outputs are typically in range [-10, 10] for untrained models
emissions = torch.randn(batch_size, seq_len, num_tags) * 5

# Create sample tags - all valid (0-8)
tags = torch.randint(0, num_tags, (batch_size, seq_len))

# Create mask - first 32 tokens are valid, rest are padding
mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
mask[:, :32] = True

print("Setup:")
print(f"  Emissions shape: {emissions.shape}")
print(f"  Tags shape: {tags.shape}")
print(f"  Mask shape: {mask.shape}")
print(f"  Emissions range: [{emissions.min():.2f}, {emissions.max():.2f}]")
print(f"  Tags range: [{tags.min()}, {tags.max()}]")
print(f"  Mask sum (valid tokens): {mask.sum()}")

# Compute loss
log_likelihood = crf(emissions, tags, mask)
loss = -log_likelihood

print(f"\nResults:")
print(f"  Log likelihood: {log_likelihood.item():.4f}")
print(f"  Loss: {loss.item():.4f}")

if abs(loss.item()) > 1000:
    print(f"\n⚠️  WARNING: Loss is very large!")
else:
    print(f"\n✅ Loss is reasonable")

# Now test with VERY large emissions (to see if this causes overflow)
print("\n" + "="*60)
print("Testing with very large emissions:")
emissions_large = torch.randn(batch_size, seq_len, num_tags) * 100
log_likelihood_large = crf(emissions_large, tags, mask)
loss_large = -log_likelihood_large
print(f"  Emissions range: [{emissions_large.min():.2f}, {emissions_large.max():.2f}]")
print(f"  Loss: {loss_large.item():.4f}")

# Test with extremely large emissions
print("\n" + "="*60)
print("Testing with extremely large emissions:")
emissions_huge = torch.randn(batch_size, seq_len, num_tags) * 1000
log_likelihood_huge = crf(emissions_huge, tags, mask)
loss_huge = -log_likelihood_huge
print(f"  Emissions range: [{emissions_huge.min():.2f}, {emissions_huge.max():.2f}]")
print(f"  Loss: {loss_huge.item():.2e}")
