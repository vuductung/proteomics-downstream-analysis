import os
import ipykernel
from IPython.core.getipython import get_ipython

def is_jupyter_notebook():
    return get_ipython() is not None and isinstance(get_ipython(), ipykernel.zmqshell.ZMQInteractiveShell)

def format_ytick_label(ytick):
    max_width = 15
    words = ytick.split()
    res = ''
    line = ''
    for w in words:
        line += (w + ' ')
        if len(line) > max_width:
            res += line.strip() + '<br>'
            line = ''
            
    res += line.strip()
    return res

def float_string_split(data):
    return data.select_dtypes(float), data.select_dtypes('string')

