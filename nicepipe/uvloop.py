'''Side-effect import to use uvloop'''
# uvloop only available on unix platform
try:
    import uvloop  # type: ignore
    uvloop.install()
# means we on windows
except ModuleNotFoundError:
    pass
