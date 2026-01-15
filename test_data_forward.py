#!/usr/bin/env python3
"""Test actual data loading and model forward pass"""

import torch
import sys
import os
sys.path.insert(0, '/Users/ez/ecom_bert_crf/BERT-CRF-EVAL')

from transformers import BertConfig, BertTokenizer
from models.bert_for_ner import BertCrfForNer
from processors.ner_seq import ner_processors as processors, convert_examples_to_features

# Setup
data_dir = '/Users/ez/ecom_bert_crf/BERT-CRF-EVAL/datasets/cner'
model_path = '/Users/ez/ecom_bert_crf/BERT-CRF-EVAL/bert-base-chinese'

# Check if model exists
if not os.path.exists(model_path):
    print(f"ERROR: Model not found at {model_path}")
    print("Please download bert-base-chinese first")
    sys.exit(1)

# Labels
label_list = ['B-HCCX', 'B-HPPX', 'B-MISC', 'B-XH', 'I-HCCX', 'I-HPPX', 'I-MISC', 'I-XH', 'O']
num_labels = len(label_list)

print(f"Number of labels: {num_labels}")
print(f"Labels: {label_list}")

# Load processor and get examples
processor = processors['cner']()
train_examples = processor.get_train_examples(data_dir)
print(f"\nLoaded {len(train_examples)} training examples")

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)

# Convert first example
features = convert_examples_to_features(
    examples=train_examples[:1],
    tokenizer=tokenizer,
    label_list=label_list,
    max_seq_length=128,
    cls_token=tokenizer.cls_token,
    sep_token=tokenizer.sep_token,
    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
)

print(f"\nFirst feature:")
print(f"  input_ids length: {len(features[0].input_ids)}")
print(f"  label_ids length: {len(features[0].label_ids)}")
print(f"  label_ids range: {min(features[0].label_ids)} to {max(features[0].label_ids)}")
print(f"  unique labels: {set(features[0].label_ids)}")

# Create tensors
input_ids = torch.tensor([features[0].input_ids], dtype=torch.long)
attention_mask = torch.tensor([features[0].input_mask], dtype=torch.long)
token_type_ids = torch.tensor([features[0].segment_ids], dtype=torch.long)
labels = torch.tensor([features[0].label_ids], dtype=torch.long)

print(f"\nTensor shapes:")
print(f"  input_ids: {input_ids.shape}")
print(f"  attention_mask: {attention_mask.shape}")
print(f"  labels: {labels.shape}")

# Load model
config = BertConfig.from_pretrained(model_path, num_labels=num_labels)
model = BertCrfForNer.from_pretrained(model_path, config=config)
model.eval()

print(f"\nModel loaded successfully")
print(f"  CRF num_tags: {model.crf.num_tags}")
print(f"  Classifier output size: {model.classifier.out_features}")

# Forward pass
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=labels
    )
    
    loss = outputs[0]
    logits = outputs[1]
    
    print(f"\nForward pass results:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: {logits.min().item():.4f} to {logits.max().item():.4f}")
    print(f"  Logits mean: {logits.mean().item():.4f}")
    print(f"  Logits std: {logits.std().item():.4f}")
    print(f"  Loss: {loss.item()}")
    
    if loss.item() > 1000:
        print(f"\n⚠️  WARNING: Loss is very large! ({loss.item():.2e})")
    else:
        print(f"\n✅ Loss looks reasonable")
