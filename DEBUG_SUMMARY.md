# BERT-CRF Training Issues - Debug Summary

## Issues Found and Fixed

### Issue 1: Label Mismatch (CRITICAL)
**Problem:** The `CnerProcessor.get_labels()` method was returning labels for a completely different dataset.
- **Expected labels:** `['B-HCCX', 'B-HPPX', 'B-MISC', 'B-XH', 'I-HCCX', 'I-HPPX', 'I-MISC', 'I-XH', 'O']` (9 labels)
- **Actual labels returned:** `['X', 'B-CONT', 'B-EDU', 'B-LOC', ...]` (23 labels)
- **Impact:** Caused astronomical loss values (10^23) due to label index mismatches
- **Fix:** Updated `processors/ner_seq.py` line 187 to return correct labels

### Issue 2: Attention Mask Data Type
**Problem:** Attention mask was being passed as `long` tensor instead of `bool`
- **Impact:** Caused numerical instability and deprecation warnings
- **Fix:** Added `.bool()` conversion in `models/bert_for_ner.py` before passing to CRF

### Issue 3: Deprecated uint8 Usage
**Problem:** CRF was using deprecated `torch.uint8` dtype for masks
- **Impact:** PyTorch deprecation warnings and potential future compatibility issues
- **Fix:** Updated `models/layers/crf.py` to use `torch.bool` instead of `torch.uint8`

### Issue 4: CRF Parameter Initialization
**Problem:** `init_weights()` from parent class may have been corrupting CRF parameters
- **Impact:** Potentially causing large loss values (10^18) due to improperly initialized transition matrices
- **Fix:** Added explicit `self.crf.reset_parameters()` call after `init_weights()` in `models/bert_for_ner.py`

## Files Modified

1. **processors/ner_seq.py**
   - Line 187: Fixed `get_labels()` to return correct e-commerce labels

2. **models/bert_for_ner.py**
   - Line 54: Added explicit CRF parameter reset
   - Line 64: Added attention mask bool conversion

3. **models/layers/crf.py**
   - Line 77: Changed mask dtype from `uint8` to `bool`
   - Line 79: Changed mask conversion from `.byte()` to `.bool()`
   - Line 124: Changed mask dtype from `uint8` to `bool`
   - Line 127: Changed mask conversion from `.byte()` to `.bool()`

## Expected Behavior After Fixes

- ✅ Loss values should start around 2-10 (not 10^18 or 10^23)
- ✅ Loss should decrease smoothly during training
- ✅ No `inf` or `nan` values
- ✅ No deprecation warnings about uint8
- ✅ Model should converge to reasonable F1 scores

## Testing Instructions

1. Pull latest changes on remote server:
   ```bash
   cd /content/BERT-CRF-EVAL
   git pull
   ```

2. Delete cached features:
   ```bash
   rm /content/BERT-CRF-EVAL/datasets/cner/cached_*
   ```

3. Run training and verify loss values are reasonable (< 100 initially)

## Root Cause Analysis

The primary issue was **label mismatch** - the processor was configured for a different dataset (CNER with labels like CONT, EDU, LOC) while the actual data used e-commerce labels (HCCX, HPPX, MISC, XH). This caused the model to try to predict label indices that didn't exist in its output layer, leading to catastrophic numerical errors in the CRF loss computation.

The secondary issues (mask dtype, CRF initialization) were contributing factors that could cause numerical instability even with correct labels.
