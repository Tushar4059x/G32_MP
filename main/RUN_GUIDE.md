# How to Run the Federated Learning Experiment

## Setup

1. **Install dependencies:**
```bash
cd trial-1
pip install -e .
```

## Run Training

2. **Start the 20-round training:**
```bash
flwr run .
```

This will:
- Train for 20 rounds with 5 hospitals
- Each hospital has TRUE 50-50 balanced data (equal NORMAL and PNEUMONIA cases)
- Display clear round numbers and metrics after each round
- Save metrics to `metrics_history.json` automatically

## View Results

3. **After training completes, plot the results:**
```bash
python plot_results.py
```

This will:
- Generate a comprehensive 6-panel visualization showing:
  - Accuracy over rounds
  - Precision vs Recall
  - F1-Score progression
  - Specificity improvement
  - AUC-ROC score
  - Loss reduction
- Save the plot as `training_results.png`
- Display final metrics summary
- Show improvement from Round 1 to Round 20

## What to Expect

### During Training
You'll see output like:
```
================================================================================
🔄 FEDERATED LEARNING - ROUND 1
================================================================================
📊 Training Phase: 5 hospitals completed local training
📉 Average Training Loss: 0.3389

✓ Client 0: 268 NORMAL + 268 PNEUMONIA = 536 total (50-50 balanced)
✓ Client 1: 268 NORMAL + 268 PNEUMONIA = 536 total (50-50 balanced)
...

📈 Evaluation Phase: Aggregated metrics from 5 hospitals
--------------------------------------------------------------------------------
  Loss:        0.8979
  Accuracy:    0.7500 (75.00%)
  Precision:   0.7800
  Recall:      0.9500 ⭐
  F1-Score:    0.8571
  Specificity: 0.5500
  AUC-ROC:     0.9000
--------------------------------------------------------------------------------
================================================================================
```

### After Training
The plot will show:
- How metrics improved over 20 rounds
- Whether the model converged
- Final performance metrics
- Comparison between Round 1 and Round 20

## Files Generated

- `metrics_history.json` - Raw metrics data from all rounds
- `training_results.png` - Visualization of training progress

## Key Improvements Made

1. ✅ **TRUE 50-50 Data Split**: Each hospital now has exactly equal NORMAL and PNEUMONIA cases
2. ✅ **Clear Round Logging**: Each round is clearly marked with round number and phase
3. ✅ **Automatic Metrics Tracking**: All metrics saved to JSON after each round
4. ✅ **Comprehensive Plotting**: 6 different metric visualizations
5. ✅ **Summary Statistics**: Final metrics and improvement summary

## Troubleshooting

If you see imbalanced data like "268 NORMAL, 775 PNEUMONIA":
- The old code is still running
- Make sure you've reinstalled: `pip install -e .`
- Check that you're in the trial-1 directory

If plotting fails:
- Make sure training completed at least 1 round
- Check that `metrics_history.json` exists
- Verify matplotlib is installed: `pip install matplotlib`
