# blender plugin for making use of deep illumination result.
# the model could be trained by raw EEVEE inputs
# and attempting to output the result like cycles.

import bpy
import pycompositor
from PIL import Image
import requests, io, os

url = 'http://localhost:5000/rawupload'

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

    albedo_img, albedo_meta = pycompositor.array_to_pil(images[0])
    direct_img, direct_meta = pycompositor.array_to_pil(images[1])
    normal_img, normal_meta = pycompositor.array_to_pil(images[2])
    depth_img, depth_meta = pycompositor.array_to_pil(images[3])

    albedo = io.BytesIO()
    albedo_img.save(albedo, format='PNG')
    normal = io.BytesIO() 
    normal_img.save(normal, format='PNG')
    depth = io.BytesIO()
    depth_img.save(depth, format='PNG')
    direct = io.BytesIO()
    direct_img.save(direct, format='PNG')

    files = {
        'albedo': ('albedo.png', albedo.getvalue(), 'image/png'),
        'normal': ('normal.png', normal.getvalue(), 'image/png'),
        'depth': ('depth.png', depth.getvalue(), 'image/png'),
        'direct': ('direct.png', direct.getvalue(), 'image/png'),
    }

    r = requests.post(url, files=files)
    result_img = Image.open(io.BytesIO(r.content))
    result_arr = pycompositor.pil_to_array(result_img, (1.0, 1.0))
    pycompositor.array_to_pil(result_arr)[0].save(os.path.join(basedir,"result.png"))  # normal save here.

    # maybe there is some defact that we need to do color correction for displaying.
    result_arr[...,0:3] = result_arr[...,0:3] ** 2.2    # gamma correction=0.45

    # meta is defined as (maxRgb, maxAlpha)
    # see 2.79/scripts/modules/pycompositor.py
    out[:] = result_arr