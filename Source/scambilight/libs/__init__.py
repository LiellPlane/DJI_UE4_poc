import sys
import os
from pathlib import Path
import imp
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
#sys.path.append(str(Path(parent).resolve().parents[0]) + "\\lumotag\\factory.py")
module_name = imp.load_source('factory', str(Path(parent).resolve().parents[0]) + "\\lumotag\\factory.py")
module_name = imp.load_source('img_processing', str(Path(parent).resolve().parents[0]) + "\\lumotag\\img_processing.py")