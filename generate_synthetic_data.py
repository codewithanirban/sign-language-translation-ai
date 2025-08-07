import numpy as np
import os

# Generate synthetic ASL keypoint data
np.random.seed(42)

# ASL classes: A-Z, space, nothing, del
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
               'space', 'nothing', 'del']
num_classes = len(class_names)

# Generate realistic hand keypoint patterns for each class
def generate_hand_keypoints(gesture_type):
    """Generate synthetic hand keypoints for different ASL gestures"""
    keypoints = np.zeros(63)  # 21 points * 3 coordinates
    
    if gesture_type == 'A':  # Fist with thumb on side
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
        
    elif gesture_type == 'B':  # Flat hand
        # Thumb
        keypoints[0:3] = [0.1, 0.5, 0.0]
        keypoints[3:6] = [0.15, 0.45, 0.0]
        keypoints[6:9] = [0.2, 0.4, 0.0]
        keypoints[9:12] = [0.25, 0.35, 0.0]
        
        # All fingers extended
        for i in range(4):  # 4 fingers
            base_idx = 12 + i * 9
            keypoints[base_idx:base_idx+3] = [0.3 + i*0.05, 0.6, 0.0]      # Tip
            keypoints[base_idx+3:base_idx+6] = [0.35 + i*0.05, 0.5, 0.0]   # PIP
            keypoints[base_idx+6:base_idx+9] = [0.4 + i*0.05, 0.4, 0.0]    # MCP
        
        # Wrist
        keypoints[48:51] = [0.3, 0.0, 0.0]
        
    elif gesture_type == 'C':  # Curved hand
        # Thumb
        keypoints[0:3] = [0.1, 0.5, 0.0]
        keypoints[3:6] = [0.15, 0.45, 0.0]
        keypoints[6:9] = [0.2, 0.4, 0.0]
        keypoints[9:12] = [0.25, 0.35, 0.0]
        
        # Curved fingers
        for i in range(4):
            base_idx = 12 + i * 9
            keypoints[base_idx:base_idx+3] = [0.3 + i*0.05, 0.4, 0.0]      # Tip
            keypoints[base_idx+3:base_idx+6] = [0.35 + i*0.05, 0.5, 0.0]   # PIP
            keypoints[base_idx+6:base_idx+9] = [0.4 + i*0.05, 0.4, 0.0]    # MCP
        
        # Wrist
        keypoints[48:51] = [0.3, 0.0, 0.0]
        
    else:  # Random gesture for other letters
        # Generate random but realistic hand positions
        for i in range(21):
            x = 0.2 + 0.3 * np.random.random()
            y = 0.1 + 0.4 * np.random.random()
            z = -0.1 + 0.2 * np.random.random()
            keypoints[i*3:(i+1)*3] = [x, y, z]
    
    return keypoints

# Generate data for each class
all_keypoints = []
all_labels = []

samples_per_class = 100  # 100 samples per class

for class_idx, class_name in enumerate(class_names):
    print(f"Generating data for class {class_name}...")
    
    for _ in range(samples_per_class):
        # Add some noise to make it more realistic
        base_keypoints = generate_hand_keypoints(class_name)
        noise = np.random.normal(0, 0.02, 63)  # Small noise
        keypoints = base_keypoints + noise
        
        all_keypoints.append(keypoints)
        all_labels.append(class_idx)

# Convert to numpy arrays
all_keypoints = np.array(all_keypoints, dtype=np.float32)
all_labels = np.array(all_labels, dtype=np.int64)

# Reshape to (num_samples, 1, 63) for static sequence
all_keypoints = all_keypoints.reshape(-1, 1, 63)

print(f"Generated {all_keypoints.shape[0]} samples")
print(f"Data shape: {all_keypoints.shape}")
print(f"Labels shape: {all_labels.shape}")
print(f"Number of classes: {num_classes}")

# Save the data
np.save('asl_keypoints.npy', all_keypoints)
np.save('asl_labels.npy', all_labels)

print("Saved synthetic ASL data to asl_keypoints.npy and asl_labels.npy") 