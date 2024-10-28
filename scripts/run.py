import subprocess as sp
import os

python_path = ".venv/bin/python3"

if(not os.path.exists(python_path)):
    python_path = ".venv/Scripts/python"

main_script_path =  " src/main.py"

command = python_path + main_script_path

# Method 1
sp.run(command, check=True, shell=True)

# Method 2
# sp.Popen([python_path, main_script_path])
