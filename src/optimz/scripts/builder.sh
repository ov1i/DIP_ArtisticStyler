#!/bin/bash
link_lib_py() {
    ./../../.venv/bin/python3 tests/test_linkage.py
}

main() {
    gcc -v
    if [ $? -ne 0 ]; then
        echo "GCC not found. Exiting with result -1"
        exit -1
    fi
    echo -e "\n\n::Build script will continue...::\n\n"

    if [ "$1" == "-l" ]; then
        echo "COMPILE of convolution lib STARTED"

        # gcc -c src/convol.c -lm -o src/convo_lib_obj.o # -> Static lib build
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then # Linux
            echo -e "\n...Linux detected continuing with the appropriate library compilation...\n"
            gcc -shared -o lib/convo_lib.so -fPIC src/convol.c -lm -mavx
        elif [[ "$OSTYPE" == "darwin"* ]]; then  # Mac OSX
            echo -e "\n...MacOS detected continuing with the appropriate library compilation...\n"
            gcc -shared -o lib/convo_lib.so -fPIC src/convol.c -lm -mfpu=neon
        else 
            echo -e "\nNo specific OS found continuing with default library compilation!\n"
            gcc -shared -o lib/convo_lib.so -fPIC src/convol.c -lm
        fi
        if [ $? -ne 0 ]; then
            echo -e "\n\nFailed to compile the library. Exiting with result -1.\n\n"
            exit -1
        fi
        echo -e "COMPILE of convolution lib FINISHED \n"
        
        echo -e "\n::Lib linkage&linkage_testing to PY started::\n"
        
        link_lib_py
        if [ $? -ne 0 ]; then
            echo -e "\n\nFailed to establish a linkage between C and PY. Exiting with result -1.\n\n"
            exit -1
        fi

        echo -e "\n\n::Lib linkage&linkage_testing done..Exiting with results 1::"
    elif [ "$1" == "-t" ]; then
        echo -e "::Testing scope is ON::\n"

        # gcc -g -lm -mavx -o tests/convo_lib_obj.o src/convol.c
        echo -e "COMPILE of convolution lib STARTED\n.."
        gcc -c src/convol.c -g -lm -o tests/convo_lib_obj.o
        if [ $? -ne 0 ]; then
            echo -e "\n\nFailed to compile the library. Exiting with result -1.\n\n"
            exit -1
        fi
        echo -e "COMPILE of convolution lib FINISHED \n"

        echo -e "\n::Library build started::\n.."
        ar rcs tests/convo_lib.a tests/convo_lib_obj.o
        if [ $? -ne 0 ]; then
            echo -e "\n\nFailed to create the static library. Exiting with result -1."
            exit -1
        fi
        echo -e "::Library built successfully::\n"
        
        echo -e "\n::Building the test program started::\n.."
        # gcc tests/run_convo_tests.c -L tests/ -l:convo_lib.a -mavx -lm -g -o tests/run_convo_tests.o
        gcc tests/run_convo_tests.c -L tests/ -l:convo_lib.a -lm -g -o tests/run_convo_tests.o
        if [ $? -ne 0 ]; then
            echo "Failed to compile the test program. Exiting with result -1."
            exit 1
        fi
        echo -e "::Building the test program finished::\n"

        echo -e "::Executing test cases::\n"
        ./tests/run_convo_tests.o
        if [ $? -ne 0 ]; then
            echo "Failed to run tests due to execution erros. Exiting with result -1."
            exit 1
        fi
        echo -e "::Execution of test cases finished::\n"

        echo "::Testing scope is OFF::"
    fi
      
    echo -e "\n\n::Build script finished executing...Exiting with result 1::\n\n"
}

if [ $# -ne 1 ]; then
    echo "Usage: $0 [option] ..."
    echo -e "OPTIONS:"
    echo -e "\t-t:\t Build and run test cases"
    echo -e "\t-l:\t Build the library and export/link it to PY"
    exit -1
fi

if [ "$1" == "-t" ] || [ "$1" == "-l" ]; then
    main $1
else
    echo "Error: Invalid OPTION"
    echo "Usage: $0 [option] ..."
    echo -e "OPTIONS:"
    echo -e "\t-t:\t Build and run test cases"
    echo -e "\t-l:\t Build the library and export/link it to PY"
    exit -1
fi
