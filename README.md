# nicepipe

Worker that receives video input & outputs analysis.

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
7. On first run, the server will generate `config.yml` in the same directory. Modifying it requires a server restart.

## Refactors Possible

- Formal lightweight Packet struct of 2 variants, immutable and mutable (mutable packets being those that hold data such as direct references to image buffers instead of copies of that buffer).
- AsyncioWorkers
  - Base concept of a worker with negligable IO times (passive data retrieval) in exchange for data-desynchronization
  - 4 variants: async (separate loop), async buffered (async queue), threaded (use thread for loop), forked (use process for loop)
  - Unified API
    - Debug API such as FPS callbacks
    - Support for usage independently or as context
    - Function to "join" loop to actively get data instead of passively
- Refactor out logging & config system to make a OmegaConf-Rich hybrid that is more portable and flexible than Hydra
- Refactor out async utils such as rate-limited loops, set_interval, set_timeout, cancel_and_join, etc
- Refactor out pip CUDA hack

## TODO

- `to_thread()` has cost. better to use for one large sync jobs than for many tiny parts in a large async job.
  - basically evaluate when async functions make sense vs sync functions, then wrap the main sync loops in jobs
  - might have to reevaluate base analysisworker structure
    - for example, allow sync io processor functions since they will be wrapped by to_thread
  - need sync version of RLLoop for sync loops
  - might ease transition if we ever choose to use non-GIL python
    - but then we might have to worry about locks...
    - the performance tho...
- When Python 3.10 becomes widely supported, use `__slots__` on all dataclasses and even some classes for free performance

## Tests Possible

- Test that AsyncioWorker has negligable IO time and non-zero output FPS
- Test logging and config merging
- All components can open & close correctly without hanging
- Integrating Sources, Analyzers and Sinks from external packages

### Live Configurability

- Level 0: Reconfigure by restarting (Y)
- Level 1: Reconfigure by closing worker, changing config & reopening (probably Y)
- Level 2: Reconfigure by closing component, changing its config & reopening (almost Y)
- Level 3: Reconfigure by changing config, internal state is live-updated. (N)

## Tips

[Postman](https://www.postman.com/) is good for testing APIs. See <https://blog.postman.com/postman-now-supports-socket-io/>.

## Quirks

Through hacks and obscurity, I utilize a secret Nvidia Wheel Repository, which just so happens to have Windows builds of the CUDA runtime (besides Linux of course). Unfortunately, specifically for cuDNN, there are only Linux wheels available. Hence the _Windows Only_ step.

Another quirk is that Nvidia didn't bother to load the DLLs in their wheels. I wrote a workaround. **`import nicepipe.cuda` must be run before `import tensorflow`!**.

The workaround is actually a side-effect import designed not only to load CUDA successfully, but also to be detectable by `PyInstaller` such that the required DLLs are copied and linked.

On Linux, while `nicepipe.cuda` works in development, and `PyInstaller` correctly detects the CUDA DLLs, the executable cannot properly be built. Even openCV cannot properly bundle in qt. My guess is my unique system configuration might be messing with `PyInstaller`'s DLL detection. It might also be that on Linux, specifying additional paths in the build command might be insufficient, and instead these paths must be specified in `LD_LIBRARY_PATH`, which can fortunately be done via `poethepoet`. That said, building for Linux isn't supported.

### Lots of Error messages on Keyboard Interrupt

I already tried to resolve most of them. There is one caused by `poethepoet` (<https://github.com/nat-n/poethepoet/issues/42>) that can be bypassed by running `python app.py` directly. See <https://stackoverflow.com/questions/70399670/how-to-shutdown-gracefully-on-keyboard-interrupt-when-an-asyncio-task-is-perform> about the rest.

I have ensured at least that the Worker can start and stop without error messages. KeyboardInterrupt is a bit more iffy.

Why is asyncio error handling so hard? That said, I've probably created quite a thorough system for it by wrapping common convenience functions. I don't get why the default asyncio functions are so intent on leaving tasks orphaned. Shutdown is 100% smooth now though, worth it.

~If only Python's API allowed me to easily treat async and sync variants of functions and iterators as the same, right now a lot of utils are awkward to write as separate variants are needed for both. is there a function to auto-convert sync stuff to async (maybe use to_thread) while ignoring async stuff?~

~Python really needs a way to be ultra flexible about kwargs~
