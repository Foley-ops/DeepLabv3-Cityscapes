import subprocess
import time

RUN_COMMAND = "python3"
RUN_FILE = "deeplabv3_VOS.py"

print(f"Starting Repeater on {RUN_FILE}...")
for i in range(10):
    start_time = time.time()
    print(f"Run #{i+1}")
    
    subprocess.run([RUN_COMMAND, RUN_FILE])
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    if elapsed_time > 60:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(f"Execution time: {minutes} minutes {seconds:.2f} seconds")
    else:
        print(f"Execution time: {elapsed_time:.2f} seconds")

    print(" ")