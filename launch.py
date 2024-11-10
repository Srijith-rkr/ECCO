import subprocess
import logging

# Set up the logging configuration
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # Custom log format
    handlers=[
        logging.StreamHandler(),  # This will print logs to the console
        logging.FileHandler('outputs.txt', mode='w')  # This will write logs to 'outputs.txt'
    ]
)

# List of commands to run
commands = [
    "python ECCO/experiments/inference.py --eval_mode edit",
    "python ECCO/experiments/inference.py --eval_mode self-refine",
    "python ECCO/experiments/inference.py --eval_mode exec-refine",
    "python ECCO/experiments/inference.py --eval_mode nl-exec-refine",
    "python ECCO/experiments/inference.py --eval_mode edit --few_shot_examples 2",
]

# Loop through each command and run it with real-time output logging
for command in commands:
    try:
        logging.info(f"Running command: {command}")
        
        # Open a subprocess to execute the command and capture stdout and stderr
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Read the output and error streams in real-time
        for stdout_line in process.stdout:
            logging.info(stdout_line.strip())  # Log stdout
        
        for stderr_line in process.stderr:
            logging.error(stderr_line.strip())  # Log stderr (error messages)

        # Wait for the process to finish and get the return code
        process.wait()
        
        if process.returncode != 0:
            logging.error(f"Command failed with return code {process.returncode}: {command}")
        else:
            logging.info(f"Command succeeded: {command}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {command}\nError: {e.stderr}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")

logging.info("All commands executed, check 'outputs.txt' for results.")