import os
import re
import subprocess
import time

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return 0
    checkpoints = [
        int(re.search(r"ppo_checkpoint(\d+)", fname).group(1))
        for fname in os.listdir(checkpoint_dir)
        if re.match(r"ppo_checkpoint\d+\.zip", fname)
    ]
    return max(checkpoints) if checkpoints else 0

def main():
    checkpoint_dir = 'box_ppo_checkpoints'
    total_saves = 100

    while True:
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint >= total_saves - 1:
            print("Training complete.")
            break

        print(f"Resuming from checkpoint index: {latest_checkpoint}")
        cmd = ["python", "box_agent_train.py", "--resume", str(latest_checkpoint)]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Training crashed. Will retry from checkpoint {latest_checkpoint} in 5 seconds...")
            time.sleep(5)

if __name__ == "__main__":
    main()
