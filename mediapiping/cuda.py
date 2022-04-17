'''Side Effect Only import to force CUDA to load'''
# Must be done like this for PyInstaller to properly detect & handle
# note how I call WinDLL(name) instead of using a forloop
# or how the name is only the basename without any path
# ^ hence the need to extend search paths
# see https://pyinstaller.org/en/stable/feature-notes.html#solution-in-pyinstaller

import os
import nvidia
from ctypes import WinDLL

PIP_NVIDIA_PATH = nvidia.__path__[0]

DLL_PATHS = [
    '.',
    'cudnn\\bin',
    'cudnn\\dll_x64',
    os.path.join(PIP_NVIDIA_PATH, 'cublas\\lib\\x64'),
    os.path.join(PIP_NVIDIA_PATH, 'cuda_runtime\\bin'),
    os.path.join(PIP_NVIDIA_PATH, 'cufft\\bin'),
    os.path.join(PIP_NVIDIA_PATH, 'curand\\bin'),
    os.path.join(PIP_NVIDIA_PATH, 'cusolver\\bin'),
    os.path.join(PIP_NVIDIA_PATH, 'cusparse\\bin')
]

for path in DLL_PATHS:
    try:
        os.add_dll_directory(os.path.abspath(path))
    except:
        pass

dlls = [
    WinDLL('cudart64_110.dll'),
    WinDLL('cublas64_11.dll'),
    WinDLL('cublasLt64_11.dll'),
    WinDLL('cufft64_10.dll'),
    WinDLL('curand64_10.dll'),
    WinDLL('cusolver64_11.dll'),
    WinDLL('cusparse64_11.dll'),
    WinDLL('cudnn64_8.dll')
]


# print('DLLs loaded: ', dlls)
