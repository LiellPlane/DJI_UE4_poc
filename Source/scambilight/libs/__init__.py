import os
from pathlib import Path
import imp
# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
#sys.path.append(str(Path(parent).resolve().parents[0]) + "\\lumotag\\factory.py")
module_name = imp.load_source('video_recorder', str(Path(parent).resolve().parents[0]) + "/lumotag/video_recorder.py")
module_name = imp.load_source('my_collections', str(Path(parent).resolve().parents[0]) + "/lumotag/my_collections.py")
module_name = imp.load_source('img_processing', str(Path(parent).resolve().parents[0]) + "/lumotag/img_processing.py")
module_name = imp.load_source('factory', str(Path(parent).resolve().parents[0]) + "/lumotag/factory.py")
module_name = imp.load_source('lumotag_utils', str(Path(parent).resolve().parents[0]) + "/lumotag/utils.py")
module_name = imp.load_source('common', str(Path(parent).resolve().parents[0]) + "/infra/scambilight/scambi/common.py")
