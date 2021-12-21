import io
from PIL import Image
import requests

url = 'http://localhost:5000/rawupload'

albedo = io.BytesIO()
Image.open('albedo.png').save(albedo, format='PNG')
normal = io.BytesIO() 
Image.open('normal.png').save(normal, format='PNG')
depth = io.BytesIO()
Image.open('depth.png').save(depth, format='PNG')
direct = io.BytesIO()
Image.open('direct.png').save(direct, format='PNG')

files = {
    'albedo': ('albedo.png', albedo.getvalue(), 'image/png'),
    'normal': ('normal.png', normal.getvalue(), 'image/png'),
    'depth': ('depth.png', depth.getvalue(), 'image/png'),
    'direct': ('direct.png', direct.getvalue(), 'image/png'),
}

r = requests.post(url, files=files)
with open('result.png', 'wb') as f:
    f.write(r.content)