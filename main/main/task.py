from collections import OrderedDict
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)

# Define the device to use (MPS for M2, CUDA if available, else CPU)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# --- 1. DEFINE THE CNN MODEL ---
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        # 1 input image channel (grayscale), 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # Calculate flattened size based on 128x128 input
        # Input 128x128 -> conv1 (124x124) -> pool (62x62)
        # -> conv2 (58x58) -> pool (29x29)
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # Output 2 classes (NORMAL, PNEUMONIA)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 29 * 29)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# --- 2. DEFINE DATA LOADING & PARTITIONING ---
def load_data(partition_id: int, num_partitions: int = 5) -> Tuple[DataLoader, DataLoader]:
    """Load the data partition for a specific client (node_id)."""
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),  # 1 channel
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load the full training dataset
    data_dir = "./chest_xray/chest_xray/train"
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Get indices for each class (0: NORMAL, 1: PNEUMONIA)
    indices_class_0 = [i for i, (img, label) in enumerate(full_dataset) if label == 0]
    indices_class_1 = [i for i, (img, label) in enumerate(full_dataset) if label == 1]

    np.random.shuffle(indices_class_0)
    np.random.shuffle(indices_class_1)

    # --- Create TRUE 50-50 balanced split across multiple clients ---
    # Balance the dataset first by taking equal numbers from each class
    if partition_id < num_partitions:
        # Determine the smaller class size to balance
        min_class_size = min(len(indices_class_0), len(indices_class_1))
        
        # Use equal numbers from each class for true 50-50 split
        balanced_c0 = indices_class_0[:min_class_size]
        balanced_c1 = indices_class_1[:min_class_size]
        
        # Now split equally among clients
        samples_per_client = min_class_size // num_partitions
        
        # Get start and end indices for this partition
        start_idx = partition_id * samples_per_client
        end_idx = start_idx + samples_per_client if partition_id < num_partitions - 1 else min_class_size
        
        # Each client gets equal samples from both classes (TRUE 50-50)
        client_c0 = balanced_c0[start_idx:end_idx]
        client_c1 = balanced_c1[start_idx:end_idx]
        client_indices = client_c0 + client_c1
        
        print(f"✓ Client {partition_id}: {len(client_c0)} NORMAL + {len(client_c1)} PNEUMONIA = {len(client_indices)} total (50-50 balanced)")
    else:
        # Invalid partition_id
        client_indices = []

    # Create the subset for the client
    trainset = Subset(full_dataset, client_indices)

    # Load the (centralized) test set
    test_dir = "./chest_xray/chest_xray/test"
    testset = datasets.ImageFolder(test_dir, transform=transform)

    # Create DataLoaders
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)

    return trainloader, testloader


# --- 3. DEFINE TRAIN FUNCTION ---
def train(
    net: Net,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device = DEVICE,
) -> float:
    """Train the network on the training set."""
    net.to(device)  # Move model to device
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    running_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        running_loss += epoch_loss
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(trainloader)}")
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


# --- 4. DEFINE TEST FUNCTION ---
def test(
    net: Net,
    testloader: DataLoader,
    device: torch.device = DEVICE,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate the network on the test set with comprehensive metrics."""
    net.to(device)  # Move model to device
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    net.eval()
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            
            # Get predictions and probabilities
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store for metric calculation
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of class 1 (PNEUMONIA)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate comprehensive metrics
    avg_loss = loss / len(testloader)
    accuracy = np.mean(all_predictions == all_labels)
    
    # Precision, Recall, F1 for PNEUMONIA class (class 1)
    precision = precision_score(all_labels, all_predictions, pos_label=1, zero_division=0)
    recall = recall_score(all_labels, all_predictions, pos_label=1, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, pos_label=1, zero_division=0)
    
    # Specificity (True Negative Rate)
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # AUC-ROC
    try:
        auc_roc = roc_auc_score(all_labels, all_probabilities)
    except ValueError:
        auc_roc = 0.0  # In case only one class is present
    
    # Print detailed metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nPneumonia Detection Metrics (Class 1):")
    print(f"  Precision:   {precision:.4f} - {precision*100:.2f}% of predicted pneumonia are correct")
    print(f"  Recall:      {recall:.4f} - {recall*100:.2f}% of actual pneumonia cases detected")
    print(f"  F1-Score:    {f1:.4f} - Harmonic mean of precision and recall")
    print(f"  Specificity: {specificity:.4f} - {specificity*100:.2f}% of normal cases correctly identified")
    print(f"  AUC-ROC:     {auc_roc:.4f} - Overall discrimination ability")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn:4d} (Correct NORMAL predictions)")
    print(f"  False Positives: {fp:4d} (NORMAL wrongly predicted as PNEUMONIA)")
    print(f"  False Negatives: {fn:4d} (PNEUMONIA wrongly predicted as NORMAL) ⚠️ CRITICAL")
    print(f"  True Positives:  {tp:4d} (Correct PNEUMONIA predictions)")
    print("="*60 + "\n")
    
    # Return metrics dictionary
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "auc_roc": float(auc_roc),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
    
    return avg_loss, metrics


# --- 5. HELPER FUNCTIONS FOR FLOWER ---
def get_weights(net: Net):
    """Get model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: Net, parameters):
    """Set model weights from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
