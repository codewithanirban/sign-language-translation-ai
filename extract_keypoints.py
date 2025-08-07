import os
import numpy as np
import cv2
from tqdm import tqdm
import mediapipe as mp

DATASET_DIR = os.path.expanduser(r'../../../../.cache/kagglehub/datasets/grassknoted/asl-alphabet/versions/1/asl_alphabet_train/asl_alphabet_train')
OUTPUT_DATA = 'asl_keypoints.npy'
OUTPUT_LABELS = 'asl_labels.npy'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Map class names to integer labels
class_names = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

all_keypoints = []
all_labels = []

for cls in tqdm(class_names, desc='Classes'):
    cls_dir = os.path.join(DATASET_DIR, cls)
    for img_name in tqdm(os.listdir(cls_dir), desc=cls, leave=False):
        img_path = os.path.join(cls_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            all_keypoints.append(keypoints)
            all_labels.append(class_to_idx[cls])
        else:
            # If no hand detected, skip
            continue

all_keypoints = np.array(all_keypoints, dtype=np.float32)
all_labels = np.array(all_labels, dtype=np.int64)
# Reshape to (num_samples, 1, 63) for static sequence
all_keypoints = all_keypoints.reshape(-1, 1, 63)
np.save(OUTPUT_DATA, all_keypoints)
np.save(OUTPUT_LABELS, all_labels)
print(f'Saved {all_keypoints.shape[0]} samples to {OUTPUT_DATA}, {OUTPUT_LABELS}')