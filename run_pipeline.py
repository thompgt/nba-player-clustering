import subprocess
import sys

def run_command(command):
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        sys.exit(1)

def main():
    print("Starting NBA Player Clustering Pipeline...\n")
    
    # 1. Preprocess
    run_command("python preprocess.py")
    
    # 2. Validate
    run_command("python validate_model.py")
    
    # 3. Test
    run_command("pytest test_preprocess.py")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
