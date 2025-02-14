# deeplabv3_resnet_model.py

import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from PIL import Image
import matplotlib.pyplot as plt
import time
import datetime
import psutil
import pynvml
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

results = []

# Suppress deprecated warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT).to(device)
model.eval()

# Download example image
import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/deeplab1.png", "deeplab1.png")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)

url2, filename2 = ("https://github.com/pytorch/hub/raw/master/images/deeplab2.png", "ground_truth.png") #validation image to train accuracy
try:
    urllib.URLopener().retrieve(url2, filename2)
except:
    urllib.request.urlretrieve(url2, filename2)


# Preprocess image
input_image = Image.open(filename).convert("RGB")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image).unsqueeze(0).to(device)


# Measure inference time
start_time = time.time()
with torch.no_grad():
    output = model(input_tensor)['out'][0]
inference_time = time.time() - start_time

# Fix data type issue for PIL conversion
output_predictions = output.argmax(0).byte().cpu().numpy()


# Load and preprocess ground truth image
gt_image = Image.open(filename2).convert('L')  # Convert to grayscale
gt_image = gt_image.resize(output_predictions.shape[::-1], Image.NEAREST)  # Resize

ground_truth = np.array(gt_image, dtype=np.uint8)

# Ensure ground truth class labels are within range
ground_truth = np.clip(ground_truth, 0, 20)

# # Compute model accuracy (placeholder using random ground truth)
# ground_truth = np.random.randint(0, 21, output_predictions.shape)
accuracy = accuracy_score(ground_truth.flatten(), output_predictions.flatten())

# Measure resource usage
cpu_usage = psutil.cpu_percent(interval=1)
mem_usage = psutil.virtual_memory().percent

gpu_usage, gpu_power = None, None
if device.type == 'cuda':
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    pynvml.nvmlShutdown()

power_usage = None
if os.path.exists('/sys/class/power_supply/battery/power_now'):
    with open('/sys/class/power_supply/battery/power_now') as f:
        power_usage = float(f.read()) / 1e6

# Display metrics
print(f"Inference Time: {inference_time:.4f} seconds")
print(f"Model Accuracy (approx): {accuracy:.4f}")
print(f"CPU Usage: {cpu_usage}%")
print(f"Memory Usage: {mem_usage}%")
if gpu_usage is not None:
    print(f"GPU Usage: {gpu_usage}%")
    print(f"GPU Power Usage: {gpu_power:.3f}W")
if power_usage is not None:
    print(f"Power Usage: {power_usage:.3f}W")

timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
benchmark_data = {
    'Timestamp': timestamp,
    'Inference Time (s)': round(inference_time, 4),
    'Model Accuracy (%)': round(accuracy * 100, 2),
    'CPU Usage (%)': cpu_usage,
    'Memory Usage (%)': mem_usage,
    'GPU Usage (%)': gpu_usage if gpu_usage is not None else 'N/A',
    'GPU Power (W)': round(gpu_power, 3) if gpu_power is not None else 'N/A',
    'Power Usage (W)': round(power_usage, 3) if power_usage is not None else 'N/A'
}

# Append results to CSV
csv_file = 'single_benchmark_results.csv'
file_exists = os.path.isfile(csv_file)

# Convert data to DataFrame
df = pd.DataFrame([benchmark_data])

# Append to CSV, creating headers if file doesn't exist
df.to_csv(csv_file, mode='a', index=False, header=not file_exists)

print(f"Benchmark data saved to {csv_file}")


# Plot results
palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
colors = (torch.arange(21)[:, None] * palette % 255).numpy().astype("uint8")
r = Image.fromarray(output_predictions).convert('P')  # Convert explicitly to mode 'P'
r.putpalette(colors)
plt.imshow(r)
plt.title("DeepLabV3-ResNet101 Segmentation")
plt.axis("off")
plt.show()