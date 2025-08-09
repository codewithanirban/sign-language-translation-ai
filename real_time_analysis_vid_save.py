import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
import torch
import mediapipe as mp
import threading    
import time
import json
from datetime import datetime
import os
from PIL import Image, ImageTk
from models.gru_attention import GRUAttentionModel
from models.bigru_attention import BiGRUAttentionModel
from models.tcn import TCN
from models.transformer import TransformerClassifier

class RealTimeSignLanguageAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Sign Language Analyzer")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.cap = None
        self.is_running = False
        self.current_model = None
        self.models = {}
        self.recording = False
        self.test_results = []
        self.video_writer = None
        self.keypoints_data = []
        self.recording_start_time = None
        
        # ASL class names
        self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                           'space', 'nothing', 'del']
        
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Load models
        self.load_models()
        
        # Create GUI
        self.create_gui()
        
    def load_models(self):
        """Load all trained models"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_paths = {
            'GRU + Attention': 'outputs/best_gru_attention.pth',
            'BiGRU + Attention': 'outputs/best_bigru_attention.pth',
            'TCN': 'outputs/best_tcn.pth',
            'Transformer': 'outputs/best_transformer.pth'
        }
        
        model_classes = {
            'GRU + Attention': GRUAttentionModel,
            'BiGRU + Attention': BiGRUAttentionModel,
            'TCN': TCN,
            'Transformer': TransformerClassifier
        }
        
        for name, path in model_paths.items():
            if os.path.exists(path):
                model = model_classes[name](input_dim=63, num_classes=29)
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                self.models[name] = model
                print(f"Loaded {name} model")
            else:
                print(f"Warning: {path} not found")
    
    def create_gui(self):
        """Create the GUI layout"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control panel (left side)
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Model selection
        ttk.Label(control_frame, text="Select Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value=list(self.models.keys())[0] if self.models else "")
        model_combo = ttk.Combobox(control_frame, textvariable=self.model_var, 
                                  values=list(self.models.keys()), state="readonly")
        model_combo.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        model_combo.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Camera controls
        ttk.Label(control_frame, text="Camera:").grid(row=2, column=0, sticky=tk.W, pady=(20, 5))
        self.start_btn = ttk.Button(control_frame, text="Start Camera", command=self.start_camera)
        self.start_btn.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.stop_btn.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Recording controls
        ttk.Label(control_frame, text="Recording:").grid(row=5, column=0, sticky=tk.W, pady=(20, 5))
        self.record_btn = ttk.Button(control_frame, text="Start Recording", command=self.toggle_recording)
        self.record_btn.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Save results
        ttk.Label(control_frame, text="Results:").grid(row=7, column=0, sticky=tk.W, pady=(20, 5))
        self.save_btn = ttk.Button(control_frame, text="Save Results", command=self.save_results)
        self.save_btn.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Clear results
        self.clear_btn = ttk.Button(control_frame, text="Clear Results", command=self.clear_results)
        self.clear_btn.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Status
        ttk.Label(control_frame, text="Status:").grid(row=10, column=0, sticky=tk.W, pady=(20, 5))
        self.status_label = ttk.Label(control_frame, text="Ready", foreground="green")
        self.status_label.grid(row=11, column=0, sticky=tk.W, pady=5)
        
        # Video frame (right side)
        video_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        video_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.video_label = ttk.Label(video_frame, text="Click 'Start Camera' to begin")
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Prediction display
        pred_frame = ttk.LabelFrame(main_frame, text="Predictions", padding="10")
        pred_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.pred_label = ttk.Label(pred_frame, text="No prediction yet", font=("Arial", 14))
        self.pred_label.grid(row=0, column=0, sticky=tk.W)
        
        # Confidence display
        self.conf_label = ttk.Label(pred_frame, text="", font=("Arial", 12))
        self.conf_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        # Results list
        results_frame = ttk.LabelFrame(main_frame, text="Test Results", padding="10")
        results_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Create Treeview for results
        columns = ('Timestamp', 'Model', 'Prediction', 'Confidence', 'Duration')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=150)
        
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
    def on_model_change(self, event=None):
        """Handle model selection change"""
        selected_model = self.model_var.get()
        if selected_model in self.models:
            self.current_model = self.models[selected_model]
            self.status_label.config(text=f"Model: {selected_model}", foreground="green")
        else:
            self.status_label.config(text="Model not found", foreground="red")
    
    def start_camera(self):
        """Start the camera feed"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            self.is_running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status_label.config(text="Camera running", foreground="green")
            
            # Start video thread
            self.video_thread = threading.Thread(target=self.update_video)
            self.video_thread.daemon = True
            self.video_thread.start()
    
    def stop_camera(self):
        """Stop the camera feed"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="Camera stopped", foreground="orange")
        self.video_label.config(text="Click 'Start Camera' to begin")
    
    # def toggle_recording(self):
    #     """Toggle recording mode"""
    #     if not self.recording:
    #         self.recording = True
    #         self.record_btn.config(text="Stop Recording")
    #         self.status_label.config(text="Recording...", foreground="red")
    #     else:
    #         self.recording = False
    #         self.record_btn.config(text="Start Recording")
    #         self.status_label.config(text="Recording stopped", foreground="orange")

    def toggle_recording(self):
        """Toggle recording mode"""
        if not self.recording:
            # Start recording
            self.recording = True
            self.record_btn.config(text="Stop Recording")
            self.status_label.config(text="Recording...", foreground="red")

            # Prepare file paths
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_video_path = f"recordings/recording_{timestamp_str}.mp4"
            self.recording_keypoints_path = f"recordings/keypoints_{timestamp_str}.json"

            os.makedirs("recordings", exist_ok=True)

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.recording_video_path, fourcc, 30, (640, 480))

            # Clear previous keypoints
            self.keypoints_data.clear()
            self.recording_start_time = time.time()

        else:
            # Stop recording
            self.recording = False
            self.record_btn.config(text="Start Recording")
            self.status_label.config(text="Recording stopped", foreground="orange")

            # Release video writer
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            # Save keypoints JSON
            with open(self.recording_keypoints_path, "w") as f:
                json.dump(self.keypoints_data, f, indent=2)

            messagebox.showinfo("Saved", 
                f"Video saved to: {self.recording_video_path}\n"
                f"Keypoints saved to: {self.recording_keypoints_path}")
    
    def extract_keypoints(self, frame):
        """Extract hand keypoints from frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
            return np.array(keypoints, dtype=np.float32), hand_landmarks
        return None, None
    
    def predict(self, keypoints):
        """Make prediction using current model"""
        if self.current_model is None:
            return None, 0.0
        
        # Reshape for model input (batch, seq_len, features)
        input_data = torch.tensor(keypoints.reshape(1, 1, 63), dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.current_model(input_data)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, prediction].item()
        
        return prediction, confidence
    
    def update_video(self):
        """Update video feed in separate thread"""
        while self.is_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Extract keypoints
                    keypoints, hand_landmarks = self.extract_keypoints(frame)
                    # Save video frame if recording
                    if self.recording and self.video_writer:
                        resized_frame = cv2.resize(frame, (640, 480))
                        self.video_writer.write(resized_frame)

                    # Save keypoints if recording
                    if self.recording and keypoints is not None:
                        self.keypoints_data.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                            "keypoints": keypoints.tolist()
                        })
                    # Draw hand landmarks
                    if hand_landmarks:
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        
                        # Make prediction
                        prediction, confidence = self.predict(keypoints)
                        
                        if prediction is not None:
                            predicted_class = self.class_names[prediction]
                            
                            # Update prediction display
                            self.root.after(0, lambda: self.pred_label.config(
                                text=f"Prediction: {predicted_class}"))
                            self.root.after(0, lambda: self.conf_label.config(
                                text=f"Confidence: {confidence:.3f}"))
                            
                            # Record if recording is active
                            if self.recording:
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                model_name = self.model_var.get()
                                duration = time.time()  # You can calculate actual duration
                                
                                result = {
                                    'timestamp': timestamp,
                                    'model': model_name,
                                    'prediction': predicted_class,
                                    'confidence': confidence,
                                    'duration': duration
                                }
                                
                                self.test_results.append(result)
                                self.root.after(0, lambda: self.add_result_to_tree(result))
                    
                    # Convert frame for tkinter
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_pil = frame_pil.resize((640, 480), Image.Resampling.LANCZOS)
                    frame_tk = ImageTk.PhotoImage(frame_pil)
                    
                    # Update video label
                    self.root.after(0, lambda: self.video_label.config(image=frame_tk))
                    self.root.after(0, lambda: setattr(self, 'current_frame', frame_tk))
            
            time.sleep(0.03)  # ~30 FPS
    
    def add_result_to_tree(self, result):
        """Add result to the results treeview"""
        self.results_tree.insert('', 'end', values=(
            result['timestamp'],
            result['model'],
            result['prediction'],
            f"{result['confidence']:.3f}",
            f"{result['duration']:.3f}s"
        ))
    
    def save_results(self):
        """Save test results to file"""
        if not self.test_results:
            messagebox.showwarning("Warning", "No results to save")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            messagebox.showinfo("Success", f"Results saved to {filename}")
    
    def clear_results(self):
        """Clear all results"""
        self.test_results.clear()
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        messagebox.showinfo("Info", "Results cleared")
    
    def on_closing(self):
        """Handle window closing"""
        self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = RealTimeSignLanguageAnalyzer(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
