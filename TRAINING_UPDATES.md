# Training Configuration Updates - Summary

## ✅ Changes Completed

### 1. **Save Frequency Changed: 2500 → 1000**
   - **File**: `configs/train_diffusion.yaml`
   - **Lines Changed**: 
     - `save_freq: 2500` → `save_freq: 1000`
     - `val_freq: 2500` → `val_freq: 1000`
   - **Result**: Model now saves checkpoints every **1000 training steps** instead of 2500

### 2. **Added `--output` Directory Argument**
   - **File**: `scripts/train_diffusion.py`
   - **New Argument**: `--output` (optional string)
   - **Purpose**: Specify custom output directory for all checkpoints and logs
   - **Default Behavior**: If not specified, uses `logs/` and `models/` as before

### 3. **Directory Structure Updates**
   
   **With `--output` specified:**
   ```
   <output>/
   ├── checkpoints/
   │   ├── checkpoint_1000.pt
   │   ├── checkpoint_2000.pt
   │   ├── checkpoint_3000.pt
   │   ├── checkpoint_latest.pt  (always points to most recent)
   │   └── ...
   └── logs/
       └── <timestamp>/
           └── (TensorBoard logs, training.log)
   ```
   
   **Without `--output` (default):**
   ```
   models/
   ├── checkpoint_1000.pt
   ├── checkpoint_2000.pt
   └── ...
   
   logs/
   └── <timestamp>/
       └── (TensorBoard logs, training.log)
   ```

### 4. **Enhanced Checkpoint Saving**
   - Now saves **two types** of checkpoints:
     1. **Step-numbered**: `checkpoint_1000.pt`, `checkpoint_2000.pt`, etc.
     2. **Latest**: `checkpoint_latest.pt` (always overwritten with newest)
   - All checkpoints include: model weights, EMA weights, optimizer state, training step, and config

## 🚀 Usage Examples

### **Kaggle Environment**
```bash
python scripts/train_diffusion.py \
    --config configs/train_diffusion.yaml \
    --output /kaggle/working/results \
    --data-root /kaggle/input/preprocessed-data \
    --use-wandb \
    --wandb-project topobrain
```

### **Local Environment**
```bash
python scripts/train_diffusion.py \
    --config configs/train_diffusion.yaml \
    --output ./experiments/run_001 \
    --data-root ./preprocessed
```

### **Default Behavior (No Output Specified)**
```bash
python scripts/train_diffusion.py \
    --config configs/train_diffusion.yaml
# Uses default: logs/ and models/
```

### **Dry Run Test**
```bash
python scripts/train_diffusion.py \
    --config configs/train_diffusion.yaml \
    --output /kaggle/working/results \
    --dry-run
# Runs 10 iterations to verify everything works
```

## 📋 Verification Checklist

✅ `save_freq` changed from 2500 → 1000 in config  
✅ `val_freq` changed from 2500 → 1000 in config  
✅ `--output` argument added to argument parser  
✅ Log directory uses `--output` when specified  
✅ Checkpoint directory uses `--output` when specified  
✅ Directories created automatically if they don't exist  
✅ Both numbered and "latest" checkpoints saved  
✅ Default behavior maintained when `--output` not specified  
✅ Compatible with Kaggle paths (`/kaggle/working/...`)  
✅ No syntax errors in modified code  

## 🔍 Files Modified

1. **`configs/train_diffusion.yaml`**
   - Changed `save_freq: 2500` → `save_freq: 1000`
   - Changed `val_freq: 2500` → `val_freq: 1000`

2. **`scripts/train_diffusion.py`**
   - Added `--output` argument to parser (line ~66)
   - Updated logging directory logic (lines ~85-95)
   - Updated checkpoint saving logic (lines ~250-277)
   - Added `checkpoint_latest.pt` saving

## 💡 Benefits

1. **More Frequent Checkpoints**: Save every 1000 steps instead of 2500
2. **Kaggle-Compatible**: Works seamlessly with Kaggle's file system
3. **Organized Output**: All artifacts in one directory
4. **Resume-Friendly**: Always have a `checkpoint_latest.pt` to resume from
5. **Flexible**: Can specify custom output path or use defaults
6. **Backward Compatible**: Works exactly as before if `--output` not specified

## 🐛 Testing

Run the test script to verify:
```bash
python test_output_args.py
```

Expected output:
- ✅ Argument parsing test
- ✅ Directory structure test
- ✅ Default behavior test
- ✅ All tests passed!

## 📝 Notes

- Checkpoints include full training state (model, optimizer, EMA, step, config)
- TensorBoard logs go to `<output>/logs/<timestamp>/`
- Each checkpoint is ~200-500MB depending on model size
- `checkpoint_latest.pt` always points to most recent save
