import bpy
import pycompositor
import requests

def on_main(context):
    """on_main() is called in the main thread and can be used to access Blender and other libraries in a 
        thread-safe manner. It is possible to add extra data to the context object and use it later
        in the on_async()."""
    pass

def on_async(context):
    """on_async() is called in background thread after on_main() and should be used for processing 
        that takes a long time in order to avoid blocking Blender UI. Do not touch anything that is not
        thread safe from this function."""
    images = context["images"]  # inputs are 
    inputs = context["inputs"]
    outputs = context["outputs"]
    out = outputs[0]

    out[:] = images[0]