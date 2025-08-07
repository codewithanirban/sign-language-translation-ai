import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from data_utils import load_data_npy, prepare_dataloaders
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

def train_model(model_name, model_class, train_loader, val_loader, test_loader, num_classes, device, epochs=20):
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    model = model_class(input_dim=63, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc, best_model_path = 0, None
    
    start_time = time.time()
    
    for epoch in range(epochs):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_f1 = eval_model(model, val_loader, criterion, device)
        
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f} F1: {v_f1:.4f}")
        
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_model_path = f'outputs/best_{model_name}.pth'
            torch.save(model.state_dict(), best_model_path)
    
    elapsed = time.time() - start_time
    
    # Test evaluation
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_f1 = eval_model(model, test_loader, criterion, device)
    
    # Inference time
    model.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x.to(device)
        start = time.time()
        for _ in range(100):
            _ = model(x)
        end = time.time()
        inf_time = (end - start) / (100 * x.size(0)) * 1000  # ms/sample
    
    results = {
        'model_name': model_name,
        'train_time': elapsed,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'inference_time': inf_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    print(f"Training complete in {elapsed/60:.2f} min")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Test acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    print(f"Inference time: {inf_time:.2f} ms/sample")
    
    return results

def plot_comparison(all_results):
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, (model_name, results) in enumerate(all_results.items()):
        axes[0, 0].plot(results['train_losses'], label=f"{model_name} (train)")
        axes[0, 0].plot(results['val_losses'], label=f"{model_name} (val)")
        axes[0, 1].plot(results['train_accs'], label=f"{model_name} (train)")
        axes[0, 1].plot(results['val_accs'], label=f"{model_name} (val)")
    
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot final metrics comparison
    model_names = list(all_results.keys())
    test_accs = [all_results[name]['test_acc'] for name in model_names]
    test_f1s = [all_results[name]['test_f1'] for name in model_names]
    train_times = [all_results[name]['train_time'] for name in model_names]
    inf_times = [all_results[name]['inference_time'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, test_accs, width, label='Test Accuracy')
    axes[1, 0].bar(x + width/2, test_f1s, width, label='Test F1')
    axes[1, 0].set_title('Test Performance Comparison')
    axes[1, 0].set_xlabel('Models')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(model_names, rotation=45)
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].bar(x - width/2, train_times, width, label='Training Time (s)')
    axes[1, 1].bar(x + width/2, inf_times, width, label='Inference Time (ms)')
    axes[1, 1].set_title('Time Performance Comparison')
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Time')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Load data
    print("Loading synthetic ASL data...")
    data, labels = load_data_npy('asl_keypoints.npy', 'asl_labels.npy')
    train_loader, val_loader, test_loader = prepare_dataloaders(data, labels, batch_size=64)
    
    print(f"Data loaded: {data.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train all models
    all_results = {}
    
    for model_name, model_class in MODEL_MAP.items():
        results = train_model(
            model_name, model_class, train_loader, val_loader, test_loader, 
            num_classes=29, device=device, epochs=20
        )
        all_results[model_name] = results
    
    # Plot comparison
    plot_comparison(all_results)
    
    # Save summary results
    with open('outputs/results_summary.txt', 'w') as f:
        f.write("Model Performance Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name, results in all_results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Training Time: {results['train_time']/60:.2f} minutes\n")
            f.write(f"  Best Validation Accuracy: {results['best_val_acc']:.4f}\n")
            f.write(f"  Test Accuracy: {results['test_acc']:.4f}\n")
            f.write(f"  Test F1 Score: {results['test_f1']:.4f}\n")
            f.write(f"  Inference Time: {results['inference_time']:.2f} ms/sample\n\n")
    
    # Find best model
    best_model = max(all_results.items(), key=lambda x: x[1]['test_acc'])
    print(f"\nBest performing model: {best_model[0]} with test accuracy: {best_model[1]['test_acc']:.4f}")
    
    print("\nTraining complete! Check the 'outputs' directory for results.")

if __name__ == "__main__":
    main() 