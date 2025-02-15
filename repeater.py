import subprocess

RUN_COMMAND = "python3"
RUN_FILE = "deeplabv3_cityscapes.py"

print(f"Starting Repeater on {RUN_FILE}...")
for i in range(200):
    print(f"Run #{i}")
    subprocess.run([RUN_COMMAND, RUN_FILE])
    print(" ")