# blender plugin for making use of deep illumination result.
# the model could be trained by raw EEVEE inputs
# and attempting to output the result like cycles.

import time
import bpy
import pycompositor
from PIL import Image
import requests, io, os

url = 'http://localhost:5000/gtupload'

basedir = os.path.dirname(bpy.data.filepath)
if not basedir:
    raise Exception("Blend file is not saved")

def on_main(context):
    """on_main() is called in the main thread and can be used to access Blender and other libraries in a 
        thread-safe manner. It is possible to add extra data to the context object and use it later
        in the on_async()."""
    pass

def on_async(context):
    """on_async() is called in background thread after on_main() and should be used for processing 
        that takes a long time in order to avoid blocking Blender UI. Do not touch anything that is not
        thread safe from this function."""
    images = context["images"]  # inputs are albedo, direct, normal, depth
    inputs = context["inputs"]
    outputs = context["outputs"]
    out = outputs[0]

    gt_img, gt_meta = pycompositor.array_to_pil(images[0])
    gt = io.BytesIO()
    gt_img.save(gt, format="PNG")

    file_suffix = "_" + str(int(time.time())) + ".png"

    files = {
        'gt': ('gt' + file_suffix, gt.getvalue(), "image/png"),
    }

    r = requests.post(url, files=files)

    # meta is defined as (maxRgb, maxAlpha)
    # see 2.79/scripts/modules/pycompositor.py
    out[:] = images[0]