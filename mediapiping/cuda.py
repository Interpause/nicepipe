'''Side Effect Only import to force CUDA to load'''
# Must be done like this for PyInstaller to properly detect & handle
# note how I call WinDLL(name) instead of using a forloop
# or how the name is only the basename without any path
# ^ hence the need to extend search paths
# see https://pyinstaller.org/en/stable/feature-notes.html#solution-in-pyinstaller

import ctypes
from pathlib import Path
import sys
import os
import nvidia
import logging

log = logging.getLogger(__name__)

PIP_NVIDIA_PATH = Path(nvidia.__path__[0])
'''path to pip-installed nvidia folder'''

CUDA_LIBS = [
    'cuda_runtime',
    'cublas',
    'cufft',
    'curand',
    'cusolver',
    'cusparse',
    'cudnn'
]
'''names of cuda library folders'''

DLL_PATHS = []
'''paths to look for DLLs in'''

if sys.platform.startswith('linux'):
    # will not be detected by PyInstaller.
    # however, no way to extend DLL search path at runtime on Linux.
    DLL_PATHS += [PIP_NVIDIA_PATH / lib / 'lib' for lib in CUDA_LIBS]
    dlls = []
    for path in DLL_PATHS:
        if path.is_dir():
            dlls += [ctypes.CDLL(path) for path in path.glob('**/*.so.*')]

elif sys.platform.startswith('windows'):
    DLL_PATHS += [
        Path('.') / 'cudnn' / 'bin',
        Path('.') / 'cudnn' / 'dll_x64',
        PIP_NVIDIA_PATH / 'cublas' / 'lib' / 'x64'
    ]
    DLL_PATHS += [PIP_NVIDIA_PATH / lib / 'bin' for lib in CUDA_LIBS]

    for path in DLL_PATHS:
        if path.is_dir():
            os.add_dll_directory(str(path.resolve()))

    dlls = [
        ctypes.WinDLL('cudart64_110.dll'),
        ctypes.WinDLL('cublas64_11.dll'),
        ctypes.WinDLL('cublasLt64_11.dll'),
        ctypes.WinDLL('cufft64_10.dll'),
        ctypes.WinDLL('curand64_10.dll'),
        ctypes.WinDLL('cusolver64_11.dll'),
        ctypes.WinDLL('cusparse64_11.dll'),
        ctypes.WinDLL('cudnn64_8.dll')
    ]

else:
    raise ImportError(f'{__name__} does not support {sys.platform}')
