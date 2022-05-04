"""Side-effect import to force CUDA to load"""
# see https://pyinstaller.org/en/stable/feature-notes.html#solution-in-pyinstaller
# PyInstaller detects DLL loading from the compiled bytecode, hence DLLs must be
# imported in a very specific way, leading to some oddities in the code below.

import sys
import os
import ctypes
from pathlib import Path
import nvidia

PIP_NVIDIA_PATH = Path(nvidia.__path__[0])
"""path to pip-installed nvidia folder"""

CUDA_LIBS = [
    "cuda_runtime",
    "cublas",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "cudnn",
]
"""names of cuda library folders"""

DLL_PATHS = []
"""paths to look for DLLs in"""

if sys.platform.startswith("linux"):
    # will not be detected by PyInstaller.
    # however, no way to extend DLL search path at runtime on Linux.
    DLL_PATHS += [PIP_NVIDIA_PATH / lib / "lib" for lib in CUDA_LIBS]
    dlls = []
    for path in DLL_PATHS:
        if path.is_dir():
            dlls += [ctypes.CDLL(path) for path in path.glob("**/*.so.*")]

    # pyinstaller must see exact name being called in bytecode... leading to below mess:
    for i in range(999):
        try:
            i == 1 and ctypes.CDLL("libcudart.so.11.0")
            i == 2 and ctypes.CDLL("libcublas.so.11")
            i == 3 and ctypes.CDLL("libcublasLt.so.11")
            i == 4 and ctypes.CDLL("libcufft.so.10")
            i == 5 and ctypes.CDLL("libcurand.so.10")
            i == 6 and ctypes.CDLL("libcusolver.so.11")
            i == 7 and ctypes.CDLL("libcusparse.so.11")
            i == 8 and ctypes.CDLL("libcudnn.so.8")
        except:
            continue


elif sys.platform.startswith("win"):
    DLL_PATHS += [
        Path(".") / "cudnn" / "bin",
        Path(".") / "cudnn" / "dll_x64",
        PIP_NVIDIA_PATH / "cublas" / "lib" / "x64",
    ]
    DLL_PATHS += [PIP_NVIDIA_PATH / lib / "bin" for lib in CUDA_LIBS]

    for path in DLL_PATHS:
        if path.is_dir():
            os.add_dll_directory(str(path.resolve()))

    # pyinstaller must see the exact name, no leading path
    dlls = [
        ctypes.WinDLL("cudart64_110.dll"),
        ctypes.WinDLL("cublas64_11.dll"),
        ctypes.WinDLL("cublasLt64_11.dll"),
        ctypes.WinDLL("cufft64_10.dll"),
        ctypes.WinDLL("curand64_10.dll"),
        ctypes.WinDLL("cusolver64_11.dll"),
        ctypes.WinDLL("cusparse64_11.dll"),
        ctypes.WinDLL("cudnn64_8.dll"),
    ]

else:
    raise ImportError(f"{__name__} does not support {sys.platform}")
