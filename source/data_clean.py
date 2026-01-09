import os
import glob

def data_clean(path):
    files = glob.glob(path)
    for f in files:
        os.remove(f)
        