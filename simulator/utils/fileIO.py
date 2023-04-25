import datetime
import os
import glob
import sys
from pathlib import Path

def make_dir(basepath, filepath):

    # Make dir if it doesn't exist
    Path(os.path.join(basepath, filepath)).mkdir(parents=True, exist_ok=True)
    return os.path.join(basepath, filepath)


def tstamp():
    return str(datetime.datetime.now().strftime('%m%d_%H%M%S'))