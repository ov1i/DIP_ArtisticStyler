import ctypes
import platform
import os

if platform.system() == 'Windows':
    testerLIB = ctypes.CDLL(os.path.abspath('dynamic_libs/convo_lib.dll'))
else:
    testerLIB = ctypes.CDLL(os.path.abspath('dynamic_libs/convo_lib.so'))

# Section test functions
mockSumFc = testerLIB.mockSum
mockHelloWorldFc = testerLIB.mockHelloWorld

# Set functions parms data type and return type
mockSumFc.argtypes = [ctypes.c_int, ctypes.c_int]
mockSumFc.restype = ctypes.c_int

mockHelloWorldFc.argtypes = None
mockHelloWorldFc.restype = None

# Call mock functions
resMockSumFc = mockSumFc(1, 1)
print("Result of the mockSumFc is: ", resMockSumFc)

mockHelloWorldFc()
