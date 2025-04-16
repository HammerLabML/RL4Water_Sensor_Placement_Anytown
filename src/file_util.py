import os
from os import path

def check_writability(tested_file):
    writable = False
    if path.exists(tested_file):
        if path.isfile(tested_file):
            if os.access(tested_file, os.W_OK):
                writable = True
    else:
        dirname = path.dirname(tested_file)
        if path.isdir(dirname):
            if os.access(dirname, os.W_OK):
                writable = True
    if not writable:
        raise IOError(f'{tested_file} is not writable')

