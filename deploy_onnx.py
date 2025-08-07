import argparse
import os
import torch
import numpy as np
import onnx
import onnxruntime as ort
from data_utils import load_data_npy, load_data_csv
from models.gru_attention import GRUAttentionModel
from models.bigru_attention import BiGRUAttentionModel
from models.tcn import TCN
from models.transformer import TransformerClassifier
import time

MODEL_MAP = {
    'gru_attention': GRUAttentionModel,
    'bigru_attention': BiGRUAttentionModel,
    'tcn': TCN,
    'transformer': TransformerClassifier
}

def export_to_onnx(model, model_path, onnx_path, input_shape, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    dummy_input = torch.randn(*input_shape).to(device)
    torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      opset_version=12)
    print(f"Exported to {onnx_path}")

def onnx_inference(onnx_path, input_data):
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_data.astype(np.float32)})
    return outputs[0]

def benchmark_onnx(onnx_path, input_data, repeat=100):
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    start = time.time()
    for _ in range(repeat):
        _ = session.run([output_name], {input_name: input_data.astype(np.float32)})
    end = time.time()
    avg_time = (end - start) / (repeat * input_data.shape[0]) * 1000  # ms/sample
    print(f"ONNX inference time per sample: {avg_time:.2f} ms")
    return avg_time

def main():
    parser = argparse.ArgumentParser(description='Export and run ONNX inference for sign language models.')
    parser.add_argument('--model', type=str, choices=list(MODEL_MAP.keys()), required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--onnx_path', type=str, required=True)
    parser.add_argument('--data_type', type=str, choices=['npy', 'csv'], required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--labels_path', type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    # Load data (for shape and sample inference)
    if args.data_type == 'npy':
        data, _ = load_data_npy(args.data_path, args.labels_path)
    else:
        data, _ = load_data_csv(args.data_path, args.labels_path)
    input_shape = (1, data.shape[1], data.shape[2])  # (batch, seq_len, 63)
    sample_input = data[:1]

    # Export to ONNX
    model = MODEL_MAP[args.model](input_dim=63, num_classes=args.num_classes)
    export_to_onnx(model, args.model_path, args.onnx_path, input_shape, args.device)

    # ONNX inference
    outputs = onnx_inference(args.onnx_path, sample_input)
    print(f"ONNX output (logits): {outputs}")
    # Benchmark
    benchmark_onnx(args.onnx_path, data[:32])

if __name__ == "__main__":
    main()