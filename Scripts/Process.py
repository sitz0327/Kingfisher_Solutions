# master_script.py
import subprocess

def run_script(script_name):
    print(f"Running {script_name}...")
    try:
        # Use subprocess.run to execute the script
        # capture_output=True and text=True are useful for capturing output
        result = subprocess.run(['python', script_name], capture_output=True, text=True, check=True)
        print(f"Output of {script_name}:\n{result.stdout}")
        if result.stderr:
            print(f"Errors from {script_name}:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        print(f"Stderr: {e.stderr}")
        # Optionally, exit if a script fails
        exit(1)

if __name__ == "__main__":
    run_script('separateSolarDataByLocation.py')
    run_script('mergeSolarWeather.py')
    run_script('MergeDB.py')
    run_script('cleanDB.py')
    run_script('LSTM.py')