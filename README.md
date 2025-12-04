# G32_MP - Pneumonia Detection using Federated Learning

This repository contains a federated learning implementation for pneumonia detection from chest X-ray images.

## Dataset Setup

The chest X-ray dataset is not included in this repository due to size constraints. To run the project:

1. Download the pneumonia dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) or your preferred source
2. Extract the dataset to `main/chest_xray/` directory
3. The expected structure is:
   ```
   main/chest_xray/
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

- `main/` - Main project directory
  - `main/` - Source code
    - `task.py` - Model and training logic
    - `client_app.py` - Federated learning client
    - `server_app.py` - Federated learning server
  - `run_experiment.sh` - Script to run the federated learning experiment
  - `plot_results.py` - Visualization of training results
  - Documentation files (README.md, RUN_GUIDE.md, PROJECT_ANALYSIS.md)

## Getting Started

See `main/README.md` and `main/RUN_GUIDE.md` for detailed instructions on running the project.

## Requirements

- Python 3.8+
- PyTorch
- Flower (Federated Learning framework)
- See `main/pyproject.toml` for complete dependencies
