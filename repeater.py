import subprocess

for _ in range(10):
    subprocess.run(["python3", "deeplabv3_cityscapes.py"])
