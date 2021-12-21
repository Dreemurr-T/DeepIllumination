# blender plugin for make use of deep illumination result.
# the model could be trained by raw EEVEE inputs
# and attempting to output the result like cycles.

import bpy
import pycompositor
from PIL import Image
import requests

url = 'http://localhost:5000/rawupload'

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

    albedo_img, meta = pycompositor.array_to_pil(images[0])
    

    out[:] = images[0]