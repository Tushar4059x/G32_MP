# G32_MP - Pneumonia Detection using Federated Learning

This repository contains a federated learning implementation for pneumonia detection from chest X-ray images.

## Dataset Setup

The chest X-ray dataset is not included in this repository due to size constraints. To run the project:

1. Download the pneumonia dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) or your preferred source
2. Extract the dataset to `trial-1/chest_xray/` directory
3. The expected structure is:
   ```
   trial-1/chest_xray/
   ├── train/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   ├── test/
   │   ├── NORMAL/
   │   └── PNEUMONIA/
   └── val/
       ├── NORMAL/
       └── PNEUMONIA/
   ```

## Project Structure

- `trial-1/` - Main project directory
  - `trial_1/` - Source code
    - `task.py` - Model and training logic
    - `client_app.py` - Federated learning client
    - `server_app.py` - Federated learning server
  - `run_experiment.sh` - Script to run the federated learning experiment
  - `plot_results.py` - Visualization of training results
  - Documentation files (README.md, RUN_GUIDE.md, PROJECT_ANALYSIS.md)

## Getting Started

See `trial-1/README.md` and `trial-1/RUN_GUIDE.md` for detailed instructions on running the project.

## Requirements

- Python 3.8+
- PyTorch
- Flower (Federated Learning framework)
- See `trial-1/pyproject.toml` for complete dependencies
