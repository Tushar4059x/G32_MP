# Federated Learning Medical AI - Quick Start

## What This Does
This simulates 10 hospitals training a CNN model together without sharing patient data.

## Run the Simulation

```bash
# Install dependencies
pip install -e .

# Run federated learning (10 clients, 3 rounds)
flwr run .
```

## What Happens
1. Server initializes a CNN model
2. 10 simulated hospitals each get a copy
3. Each hospital trains on its private data partition
4. Hospitals send model weights (not data) back to server
5. Server averages the weights using FedAvg
6. Process repeats for 3 rounds

## Configuration
Edit `pyproject.toml` to change:
- `num-server-rounds`: Training rounds (default: 3)
- `num-supernodes`: Number of hospitals (default: 10)
- `local-epochs`: Training epochs per round (default: 1)

## Architecture
- **Model**: Simple CNN (2 conv layers + 3 FC layers)
- **Dataset**: CIFAR-10 (simulating medical images)
- **Strategy**: Federated Averaging (FedAvg)
- **Privacy**: Raw data never leaves client devices
