# Federated Learning for Pneumonia Detection - Project Analysis

## Overview

This project implements a **Federated Learning (FL)** system for pneumonia detection from chest X-ray images using the Flower framework and PyTorch. The system simulates a real-world scenario where multiple medical institutions (clients) collaboratively train a machine learning model without sharing their private patient data.

## What is Federated Learning?

Federated Learning is a machine learning approach where:
- Multiple clients (hospitals/institutions) train a shared model collaboratively
- **Raw data never leaves the client devices** - preserving privacy
- Only model updates (weights) are shared with a central server
- The server aggregates these updates to improve the global model

## Project Architecture

### Core Components

1. **Server (`server_app.py`)**: Orchestrates the federated learning process
2. **Client (`client_app.py`)**: Represents individual hospitals/institutions
3. **Task (`task.py`)**: Contains the CNN model, data loading, and training logic

### Technology Stack

- **Framework**: Flower (Federated Learning framework)
- **Deep Learning**: PyTorch
- **Dataset**: Chest X-ray images (NORMAL vs PNEUMONIA)
- **Model**: Custom CNN for binary classification
- **Device Support**: MPS (Apple Silicon), CUDA (NVIDIA GPU), or CPU

## Dataset Structure

```
chest_xray/chest_xray/
├── train/
│   ├── NORMAL/       # Normal chest X-rays
│   └── PNEUMONIA/    # Pneumonia chest X-rays
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

### Non-IID Data Distribution

The project simulates a realistic scenario with **Non-IID (Non-Independent and Identically Distributed)** data:

- **Client 0**: 80% PNEUMONIA cases, 20% NORMAL cases
- **Client 1**: 20% PNEUMONIA cases, 80% NORMAL cases

This mimics real-world situations where different hospitals may have different patient populations (e.g., a specialized respiratory hospital vs. a general hospital).

## Model Architecture

### CNN Structure (`Net` class)

```
Input: 128x128 grayscale images (1 channel)
↓
Conv1: 1 → 6 channels (5x5 kernel) + ReLU + MaxPool
↓
Conv2: 6 → 16 channels (5x5 kernel) + ReLU + MaxPool
↓
Flatten: 16 × 29 × 29 = 13,456 features
↓
FC1: 13,456 → 120 + ReLU
↓
FC2: 120 → 84 + ReLU
↓
FC3: 84 → 2 (NORMAL or PNEUMONIA)
```

**Key Features:**
- Lightweight architecture suitable for medical imaging
- Binary classification (2 output classes)
- Grayscale input (typical for X-rays)
- Standardized 128x128 input size

## Data Processing Pipeline

### Image Transformations

1. **Resize**: Scale images to 128×128 pixels
2. **Grayscale**: Convert to single channel (if not already)
3. **ToTensor**: Convert PIL image to PyTorch tensor
4. **Normalize**: Mean=0.5, Std=0.5 (standardization)

### Data Loading

- **Batch Size**: 32 images per batch
- **Shuffling**: Training data is shuffled for better learning
- **Partitioning**: Data is split between clients based on partition_id

## Federated Learning Workflow

### 1. Initialization Phase

```python
# Server initializes the global model
net = Net()
parameters = get_weights(net)
```

### 2. Training Round (Repeated for num_rounds)

**Step A: Server → Clients**
- Server sends current global model weights to selected clients
- `fraction_fit = 0.5` means 50% of clients participate per round

**Step B: Local Training**
```python
# Each client trains on their local data
train(net, trainloader, local_epochs, device)
```
- Clients train for `local_epochs` (default: 1 epoch)
- Uses Adam optimizer and CrossEntropyLoss
- Training happens on local data only

**Step C: Clients → Server**
- Clients send updated model weights back to server
- Also send metrics: training loss, number of samples

**Step D: Aggregation**
```python
# Server aggregates using FedAvg (Federated Averaging)
weighted_average(metrics)
```
- Weights are averaged based on dataset size
- Larger datasets have more influence on the global model

**Step E: Evaluation**
- Clients evaluate the global model on their test set
- Metrics: loss and accuracy

### 3. Iteration

This process repeats for `num-server-rounds` (default: 3 rounds)

## Key Functions

### Training (`train`)
- Trains the model for specified epochs
- Uses Adam optimizer (adaptive learning rate)
- Returns average training loss
- Prints loss per epoch for monitoring

### Testing (`test`)
- Evaluates model on test set
- Returns loss and accuracy
- Uses no gradient computation for efficiency

### Weight Management
- `get_weights()`: Extracts model parameters as NumPy arrays
- `set_weights()`: Loads parameters into model
- Enables seamless transfer between server and clients

## Configuration (`pyproject.toml`)

### Hyperparameters

```toml
[tool.flwr.app.config]
num-server-rounds = 3      # Number of FL rounds
fraction-fit = 0.5         # Fraction of clients per round
local-epochs = 1           # Local training epochs
```

### Federation Setup

```toml
[tool.flwr.federations.local-simulation]
options.num-supernodes = 2  # Number of simulated clients
```

## Federated Averaging (FedAvg) Strategy

The server uses **FedAvg**, the most common FL algorithm:

1. **Weighted Aggregation**: Model updates are weighted by dataset size
2. **Metrics Aggregation**: Training and evaluation metrics are also averaged
3. **Minimum Clients**: Requires at least 2 clients to be available
4. **Fraction Fit**: Only 50% of clients train per round (reduces communication)

### Why Weighted Average?

```python
def weighted_average(metrics):
    total_samples = sum(num_samples for num_samples, _ in metrics)
    for key in metric_keys:
        weighted_sum = sum(num_samples * m[key] for num_samples, m in metrics)
        aggregated[key] = weighted_sum / total_samples
```

Clients with more data have proportionally more influence on the global model, which typically leads to better performance.

## Privacy Preservation

### What is Shared
- ✅ Model weights (parameters)
- ✅ Aggregated metrics (loss, accuracy)
- ✅ Number of training samples

### What is NOT Shared
- ❌ Raw X-ray images
- ❌ Patient data
- ❌ Individual predictions
- ❌ Gradients (only final weights)

This ensures patient privacy while enabling collaborative learning.

## Running the Project

### Installation
```bash
cd trial-1
pip install -e .
```

### Execution
```bash
flwr run .
```

### What Happens
1. Flower initializes the simulation environment
2. Creates 2 virtual SuperNodes (clients)
3. Server distributes initial model
4. Clients train for 3 rounds
5. Final global model is produced
6. Metrics are logged throughout

## Device Optimization

The code automatically selects the best available device:

```python
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")      # Apple Silicon (M1/M2/M3)
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")     # NVIDIA GPU
else:
    DEVICE = torch.device("cpu")      # CPU fallback
```

## Advantages of This Approach

1. **Privacy**: Patient data remains at each institution
2. **Collaboration**: Hospitals benefit from collective knowledge
3. **Scalability**: Easy to add more clients
4. **Realistic**: Non-IID data mimics real-world scenarios
5. **Flexibility**: Can switch between simulation and deployment

## Potential Improvements

1. **Model Architecture**: Use ResNet or EfficientNet for better accuracy
2. **Data Augmentation**: Add rotation, flipping for robustness
3. **More Clients**: Simulate 10+ hospitals
4. **Differential Privacy**: Add noise to weights for stronger privacy
5. **Secure Aggregation**: Encrypt model updates
6. **Class Imbalance**: Handle imbalanced datasets better
7. **Cross-Validation**: Use validation set more effectively

## Real-World Deployment

To deploy in production:

1. **Switch to Deployment Engine**: Use actual distributed clients
2. **Enable TLS**: Secure communications between server and clients
3. **Authentication**: Verify client identities
4. **Monitoring**: Track model performance over time
5. **Compliance**: Ensure HIPAA/GDPR compliance

## Conclusion

This project demonstrates a practical implementation of federated learning for medical imaging. It shows how multiple healthcare institutions can collaboratively train a pneumonia detection model while maintaining patient privacy—a critical requirement in healthcare AI.

The use of Flower framework makes the code clean, modular, and production-ready, while PyTorch provides the deep learning capabilities needed for image classification.
