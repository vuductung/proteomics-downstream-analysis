import os
import ipykernel
from IPython.core.getipython import get_ipython

def is_jupyter_notebook():
    return get_ipython() is not None and isinstance(get_ipython(), ipykernel.zmqshell.ZMQInteractiveShell)