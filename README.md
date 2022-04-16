# mediapiping

Coming up with a better name soon! Nicepipe?

## Setup Steps

1. Install [Poetry](https://python-poetry.org/docs/), a modern dependency management & packaging solution built on top of `pip`.
2. (_Windows Only_) Go to <https://developer.nvidia.com/rdp/cudnn-download>, download cuDNN v8 for CUDA 11 & put its contents (`bin`, `include` & `lib` folders) into the `cudnn` folder.
3. Run `poetry update` to create the `venv` and install dependencies.
4. (_Optional_) Install PyTorch using `poe install-torch`.
5. Run the server using `poe dev`.
6. (_Optional_) Bundle the server for production distribution using `poe build-windows`.

## Quirks

Through hacks and obscurity, I utilize a secret Nvidia Wheel Repository, which just so happens to have Windows builds of the CUDA runtime (besides Linux of course). Unfortunately, specifically for cuDNN, there are only Linux wheels available. Hence the _Windows Only_ step.

Another quirk is that Nvidia didn't bother to load the DLLs or `.so` in their wheels. So I wrote a function to load them: `force_cuda_load()`. **It must be run before `import tensorflow`!** However, due to time constraints, said function only loads DLLs, though it should be trivial to make it load `.so` as well.
