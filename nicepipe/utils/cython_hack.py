"""must be imported as early as possible"""

import pyximport
import cython

# free performance? as long as we dont use magic dependencies it should work
pyximport.install(
    pyimport=True,
    load_py_module_on_import_failure=False,
    inplace=True,
    language_level=3,
)
CYTHON_ENABLED = cython.inline("return True")  # seemingly needed to trigger compilation
