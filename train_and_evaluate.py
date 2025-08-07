# train_and_evaluate.py
"""
Train and evaluate multiple deep learning models for sign language gesture classification from MediaPipe keypoints.
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from data_utils import load_data_npy, load_data_csv, prepare_dataloaders
from models.gru_attention import GRUAttentionModel
from models.bigru_attention import BiGRUAttentionModel
from models.tcn import TCN
from models.transformer import TransformerClassifier

MODEL_MAP = {
    'gru_attention': GRUAttentionModel,
    'bigru_attention': BiGRUAttentionModel,
    'tcn': TCN,
    'transformer': TransformerClassifier
}

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return running_loss / total, correct / total

def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            running_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return running_loss / total, acc, f1

def plot_curves(train_losses, val_losses, train_accs, val_accs, out_path):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend(); plt.title('Accuracy')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate sign language models.')
    parser.add_argument('--data_type', type=str, choices=['npy', 'csv'], required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--labels_path', type=str, required=True)
    parser.add_argument('--model', type=str, choices=list(MODEL_MAP.keys()), required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.output_dir)

    # Load data
    if args.data_type == 'npy':
        data, labels = load_data_npy(args.data_path, args.labels_path)
    else:
        data, labels = load_data_csv(args.data_path, args.labels_path)
    train_loader, val_loader, test_loader = prepare_dataloaders(data, labels, batch_size=args.batch_size)

    # Model
    model = MODEL_MAP[args.model](input_dim=63, num_classes=args.num_classes).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc, best_model_path = 0, None
    start_time = time.time()
    for epoch in range(args.epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, args.device)
        v_loss, v_acc, v_f1 = eval_model(model, val_loader, criterion, args.device)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)
        writer.add_scalar('Loss/train', t_loss, epoch)
        writer.add_scalar('Loss/val', v_loss, epoch)
        writer.add_scalar('Acc/train', t_acc, epoch)
        writer.add_scalar('Acc/val', v_acc, epoch)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} F1: {v_f1:.4f}")
        # Save best model
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_model_path = os.path.join(args.output_dir, f'best_{args.model}.pth')
            torch.save(model.state_dict(), best_model_path)
    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed/60:.2f} min. Best val acc: {best_val_acc:.4f}")
    plot_curves(train_losses, val_losses, train_accs, val_accs, os.path.join(args.output_dir, f'{args.model}_curves.png'))

    # Test evaluation
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_f1 = eval_model(model, test_loader, criterion, args.device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")
    # Inference time
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x.to(args.device)
        start = time.time()
        for _ in range(100):
            _ = model(x)
        end = time.time()
        inf_time = (end - start) / (100 * x.size(0)) * 1000  # ms/sample
    print(f"Inference time per sample (batch): {inf_time:.2f} ms")
    # Save test results
    with open(os.path.join(args.output_dir, f'{args.model}_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\nTest Acc: {test_acc:.4f}\nTest F1: {test_f1:.4f}\nInference time per sample: {inf_time:.2f} ms\n")

if __name__ == "__main__":
    main()