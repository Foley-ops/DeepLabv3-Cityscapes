import torch
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from PIL import Image
import time
import datetime
import platform
import subprocess
import psutil
import pynvml
import os
import shutil
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from pathlib import Path

# Download VOC to ./VOC2012
voc_dataset = VOCSegmentation(root='./', year='2012', image_set='train', download=True)

output_dir = './VOSsegmentation/val'
os.makedirs(output_dir, exist_ok=True)

for i, (img, target) in enumerate(voc_dataset):
    color_filename = f"voc_{i:05d}_color.png"
    label_filename = f"voc_{i:05d}_labelIds.png"
    
    img.save(os.path.join(output_dir, color_filename))
    target.save(os.path.join(output_dir, label_filename))

warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).to(device)
model.eval()

def load_and_preprocess_image(image_path):
    input_image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(input_image).unsqueeze(0).to(device)

def load_ground_truth(label_path, output_shape):
    gt_image = Image.open(label_path).convert('L')
    gt_image = gt_image.resize(output_shape[::-1], Image.NEAREST)
    return np.array(gt_image, dtype=np.uint8)

def measure_resource_usage():
    cpu_usage = psutil.cpu_percent(interval=.1)
    mem_usage = psutil.virtual_memory().percent
    gpu_usage, gpu_power = None, None
    if device.type == 'cuda':
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        pynvml.nvmlShutdown()
    return cpu_usage, mem_usage, gpu_usage, gpu_power

def safe_mean(values):
    return round(np.mean(values), 4) if values else 0.0

def save_benchmark_summary(results, csv_file='VOS_benchmark_results.csv'):
    summary = pd.DataFrame([{
        'Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Total Inferences': len(results),
        'Avg Inference Time (s)': safe_mean([r.get('Inference Time (s)', 0) for r in results]),
        'Avg Model Accuracy (%)': safe_mean([r.get('Model Accuracy (%)', 0) for r in results]),
        'Avg CPU Usage (%)': safe_mean([r.get('CPU Usage (%)', 0) for r in results]),
        'Avg Memory Usage (%)': safe_mean([r.get('Memory Usage (%)', 0) for r in results]),
        #'Avg GPU Usage (%)': safe_mean([r.get('GPU Usage (%)', 0) for r in results if r.get('GPU Usage (%)') != 'N/A']),
        'Avg GPU Power (W)': safe_mean([r.get('GPU Power (W)', 0) for r in results if r.get('GPU Power (W)') != 'N/A']),
        #'Avg Power Usage (W)': safe_mean([r.get('Power Usage (W)', 0) for r in results if r.get('Power Usage (W)') != 'N/A'])
    }])
    summary.to_csv(csv_file, mode='a', index=False, header=not os.path.isfile(csv_file))
    print(f"Run Summary saved to {csv_file}")

def process_cityscapes_images(root_dir, max_inferences=None):
    results = []
    count = 0
    for label_path in Path(root_dir).rglob('*_labelIds.png'):
        if max_inferences and count >= max_inferences:
            break
        
        color_path = str(label_path).replace('_labelIds.png', '_color.png')
        #print(f"Label path: {label_path.relative_to(root_dir)}")
        #print(f"Color path: {Path(color_path).relative_to(root_dir)}")
        if not os.path.isfile(color_path):
            continue

        input_tensor = load_and_preprocess_image(color_path)
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
        inference_time = time.time() - start_time

        output_predictions = output.argmax(0).byte().cpu().numpy()
        ground_truth = load_ground_truth(label_path, output_predictions.shape)
        accuracy = accuracy_score(ground_truth.flatten(), output_predictions.flatten())

        cpu_usage, mem_usage, gpu_usage, gpu_power = measure_resource_usage()

        results.append({
            'Inference Time (s)': inference_time,
            'Model Accuracy (%)': accuracy * 100,
            'CPU Usage (%)': cpu_usage,
            'Memory Usage (%)': mem_usage,
            'GPU Usage (%)': gpu_usage if gpu_usage is not None else 'N/A',
            'GPU Power (W)': gpu_power if gpu_power is not None else 'N/A',
        })

        print(f"Processed {label_path.stem} - Accuracy: {accuracy:.4f}, Time: {inference_time:.4f}s")
        count += 1

    save_benchmark_summary(results)

root_directory = './VOSsegmentation/val'
process_cityscapes_images(root_directory, max_inferences=10)
