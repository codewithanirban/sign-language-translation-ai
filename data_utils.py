import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split

class KeypointDataset(Dataset):
    """
    PyTorch Dataset for hand keypoint sequences and labels.
    """
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def load_data_npy(data_path, labels_path):
    """Load .npy data and labels."""
    data = np.load(data_path)
    labels = np.load(labels_path)
    return data, labels

def load_data_csv(data_path, labels_path):
    """Load .csv data and labels."""
    data = np.loadtxt(data_path, delimiter=',')
    labels = np.loadtxt(labels_path, delimiter=',')
    # Reshape if needed
    return data, labels

def prepare_dataloaders(data, labels, batch_size=64, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split data into train/val/test and return DataLoaders.
    """
    # Check if we can use stratified splitting
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_samples = np.min(counts)
    
    if min_samples >= 3:  # Need at least 3 samples per class for stratified split
        try:
            X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=val_ratio+test_ratio, random_state=seed, stratify=labels)
            val_size = int(len(X_temp) * val_ratio / (val_ratio + test_ratio))
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=len(X_temp)-val_size, random_state=seed, stratify=y_temp)
        except ValueError:
            # Fallback to random splitting if stratified fails
            X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=val_ratio+test_ratio, random_state=seed)
            val_size = int(len(X_temp) * val_ratio / (val_ratio + test_ratio))
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=len(X_temp)-val_size, random_state=seed)
    else:
        # Use random splitting for classes with insufficient samples
        X_train, X_temp, y_train, y_temp = train_test_split(data, labels, test_size=val_ratio+test_ratio, random_state=seed)
        val_size = int(len(X_temp) * val_ratio / (val_ratio + test_ratio))
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=len(X_temp)-val_size, random_state=seed)
    
    train_ds = KeypointDataset(X_train, y_train)
    val_ds = KeypointDataset(X_val, y_val)
    test_ds = KeypointDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader