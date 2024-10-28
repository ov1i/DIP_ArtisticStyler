import subprocess as sp

python_path = ".venv/bin/python3"

main_script_path =  " src/main.py"

command = python_path + main_script_path

# Method 1
sp.run(command, check=True, shell=True)

# Method 2
# sp.Popen([python_path, main_script_path])
