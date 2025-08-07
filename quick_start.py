#!/usr/bin/env python3
"""
Quick Start Demo for Real-Time Sign Language Translation
This script demonstrates the system capabilities without requiring a webcam.
"""

import numpy as np
import torch
import time
from models.gru_attention import GRUAttentionModel
from models.bigru_attention import BiGRUAttentionModel
from models.tcn import TCN
from models.transformer import TransformerClassifier

def generate_test_keypoints():
    """Generate synthetic keypoints for testing"""
    # Generate realistic hand keypoints for letter 'A'
    keypoints = np.zeros(63)
    
    # Thumb
    keypoints[0:3] = [0.1, 0.5, 0.0]   # Thumb tip
    keypoints[3:6] = [0.15, 0.45, 0.0]  # Thumb IP
    keypoints[6:9] = [0.2, 0.4, 0.0]    # Thumb MCP
    keypoints[9:12] = [0.25, 0.35, 0.0] # Thumb CMC
    
    # Index finger (closed)
    keypoints[12:15] = [0.3, 0.3, 0.0]  # Index tip
    keypoints[15:18] = [0.35, 0.25, 0.0] # Index PIP
    keypoints[18:21] = [0.4, 0.2, 0.0]   # Index MCP
    
    # Middle finger (closed)
    keypoints[21:24] = [0.35, 0.2, 0.0]  # Middle tip
    keypoints[24:27] = [0.4, 0.15, 0.0]  # Middle PIP
    keypoints[27:30] = [0.45, 0.1, 0.0]  # Middle MCP
    
    # Ring finger (closed)
    keypoints[30:33] = [0.4, 0.15, 0.0]  # Ring tip
    keypoints[33:36] = [0.45, 0.1, 0.0]  # Ring PIP
    keypoints[36:39] = [0.5, 0.05, 0.0]  # Ring MCP
    
    # Pinky (closed)
    keypoints[39:42] = [0.45, 0.1, 0.0]  # Pinky tip
    keypoints[42:45] = [0.5, 0.05, 0.0]  # Pinky PIP
    keypoints[45:48] = [0.55, 0.0, 0.0]  # Pinky MCP
    
    # Wrist
    keypoints[48:51] = [0.3, 0.0, 0.0]   # Wrist
    
    return keypoints

def test_model(model, model_name, keypoints):
    """Test a single model with given keypoints"""
    model.eval()
    input_data = torch.tensor(keypoints.reshape(1, 1, 63), dtype=torch.float32)
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_data)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, prediction].item()
    end_time = time.time()
    
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    return prediction, confidence, inference_time

def main():
    print("🚀 Real-Time Sign Language Translation - Quick Start Demo")
    print("=" * 60)
    
    # ASL class names
    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                   'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                   'space', 'nothing', 'del']
    
    # Generate test keypoints
    print("📊 Generating test keypoints (ASL letter 'A' gesture)...")
    test_keypoints = generate_test_keypoints()
    
    # Model configurations
    models = {
        'GRU + Attention': GRUAttentionModel,
        'BiGRU + Attention': BiGRUAttentionModel,
        'TCN': TCN,
        'Transformer': TransformerClassifier
    }
    
    # Load and test models
    print("\n🤖 Testing all models...")
    print("-" * 60)
    
    results = []
    
    for model_name, model_class in models.items():
        try:
            # Load model
            model = model_class(input_dim=63, num_classes=29)
            
            # Try to load trained weights
            model_path = f'outputs/best_{model_name.lower().replace(" + ", "_").replace(" ", "_")}.pth'
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(model_path))
            else:
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
            
            # Test model
            prediction, confidence, inference_time = test_model(model, model_name, test_keypoints)
            predicted_class = class_names[prediction]
            
            results.append({
                'model': model_name,
                'prediction': predicted_class,
                'confidence': confidence,
                'inference_time': inference_time
            })
            
            print(f"✅ {model_name:20} | Prediction: {predicted_class:2} | Confidence: {confidence:.3f} | Time: {inference_time:.2f}ms")
            
        except FileNotFoundError:
            print(f"⚠️  {model_name:20} | No trained model found - skipping")
        except Exception as e:
            print(f"❌ {model_name:20} | Error: {str(e)}")
    
    # Summary
    print("\n📈 Summary")
    print("-" * 60)
    
    if results:
        best_model = max(results, key=lambda x: x['confidence'])
        fastest_model = min(results, key=lambda x: x['inference_time'])
        
        print(f"🏆 Best Accuracy: {best_model['model']} ({best_model['confidence']:.3f})")
        print(f"⚡ Fastest: {fastest_model['model']} ({fastest_model['inference_time']:.2f}ms)")
        print(f"🎯 Test Gesture: ASL Letter 'A' (fist with thumb on side)")
        
        # Show all predictions
        print(f"\n📊 All Predictions:")
        for result in results:
            print(f"   {result['model']:20} → {result['prediction']} ({result['confidence']:.3f})")
    
    print("\n🎉 Demo completed!")
    print("\n💡 Next steps:")
    print("   1. Run 'python real_time_analysis.py' for live webcam testing")
    print("   2. Run 'python setup_github.py' to create GitHub repository")
    print("   3. Check 'outputs/' directory for training results and models")

if __name__ == "__main__":
    main()
