# nicepipe

Worker that receives video input & outputs predictions.

TODO: Insert planned architecture documentation.

## Setup Steps

1. Install [Poetry](https://python-poetry.org/docs/), a modern dependency management & packaging solution built on top of `pip`.
2. (_Windows Only_) Go to <https://developer.nvidia.com/rdp/cudnn-download>, download cuDNN v8 for CUDA 11 & put its contents (`bin`, `include` & `lib` folders) into the `cudnn` folder.
   - Might have to download <http://www.winimage.com/zLibDll/zlib123dllx64.zip> too. Extract its contents to `cudnn`.
   - See <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#download-windows> for more info.
3. Run `poetry update` to create the `venv` and install dependencies.
4. (_Optional_) Install PyTorch using `poe install-torch`.
5. Run the server using `poe dev`.
6. (_Optional_) Bundle the server for production distribution using `poe build-windows`.

   - If using UPX to compress the build, it may be needed to modify `.venv\Lib\site-packages\PyInstaller\compat.py` else it will timeout:

   ```py
   # .venv\Lib\site-packages\PyInstaller\compat.py
   ...
   try:
       out = proc.communicate(timeout=60)[0] # increase timeout i.e. 600
   except OSError as e:
       ...
   ```

   - Use `poe build-windows-minimal` to exclude Tensorflow & CUDA (since its not currently being used and inflates file size by a lot)

## Quirks

Through hacks and obscurity, I utilize a secret Nvidia Wheel Repository, which just so happens to have Windows builds of the CUDA runtime (besides Linux of course). Unfortunately, specifically for cuDNN, there are only Linux wheels available. Hence the _Windows Only_ step.

Another quirk is that Nvidia didn't bother to load the DLLs in their wheels. I wrote a workaround. **`import nicepipe.cuda` must be run before `import tensorflow`!**.

The workaround is actually a side-effect import designed not only to load CUDA successfully, but also to be detectable by `PyInstaller` such that the required DLLs are copied and linked.

On Linux, while `nicepipe.cuda` works in development, and `PyInstaller` correctly detects the CUDA DLLs, the executable cannot properly be built. Even openCV cannot properly bundle in qt. My guess is my unique system configuration might be messing with `PyInstaller`'s DLL detection. It might also be that on Linux, specifying additional paths in the build command might be insufficient, and instead these paths must be specified in `LD_LIBRARY_PATH`, which can fortunately be done via `poethepoet`. That said, building for Linux isn't supported.

### Lots of Error messages on Keyboard Interrupt

I already tried to resolve most of them. There is one caused by `poethepoet` (<https://github.com/nat-n/poethepoet/issues/42>) that can be bypassed by running `python app.py` directly. See <https://stackoverflow.com/questions/70399670/how-to-shutdown-gracefully-on-keyboard-interrupt-when-an-asyncio-task-is-perform> about the rest.

I have ensured at least that the Worker can start and stop without error messages. KeyboardInterrupt is a bit more iffy.
