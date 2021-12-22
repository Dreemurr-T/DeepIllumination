from time import time
import torch
import argparse
from flask import Flask, request, render_template, Response
import os
os.sys.path.append("..") 
from model_backup import G,D
from util import load_image, save_image

app = Flask(__name__)

@app.route('/')
def root():
    return render_template("index.html")

def infer():
    """
    Infer the result by plugin the inputs from save file.
    The output will be saved in result.png.
    """
    global netG

    albedo_path = 'albedo.png'
    direct_path = 'direct.png'
    normal_path = 'normal.png'
    depth_path =  'depth.png'

    albedo_image = load_image(albedo_path)
    direct_image = load_image(direct_path)
    normal_image = load_image(normal_path)
    depth_image = load_image(depth_path)

    albedo = torch.autograd.Variable(albedo_image).view(1, -1, 256, 256)
    direct = torch.autograd.Variable(direct_image).view(1, -1, 256, 256)
    normal = torch.autograd.Variable(normal_image).view(1, -1, 256, 256)
    depth = torch.autograd.Variable(depth_image).view(1, -1, 256, 256)

    start_time = time()
    out = netG(torch.cat((albedo, direct, normal, depth), 1))
    elapsed_time = time() - start_time
    global elapsed_time_str
    elapsed_time_str = "{:.3f}s".format(elapsed_time)

    out_img = out.data[0]

    save_image(out_img, 'result.png')

@app.route('/upload', methods=['post'])
def render():
    # load image
    request.files.get('albedo').save('albedo.png')
    request.files.get('normal').save('normal.png')
    request.files.get('depth').save('depth.png')
    request.files.get('direct').save('direct.png')

    infer()

    return render_template("render.html", elapsed_time=elapsed_time_str)

@app.route('/image/<img_name>')
def index(img_name):
    with open(img_name + '.png', 'rb') as f:
        image = f.read()
        resp = Response(image, mimetype="image/png")
    return resp

@app.route('/rawupload', methods=['post'])
def rawrender():
    request.files.get('albedo').save('albedo.png')
    request.files.get('normal').save('normal.png')
    request.files.get('depth').save('depth.png')
    request.files.get('direct').save('direct.png')

    infer()

    return index('result')

if __name__ == "__main__":
    print("loading model ...")
    global netG

    # read the generated checkpoint data from train.py
    G_pth_path = '../checkpoint/JiaRan/netG_model_epoch_199.pth'
    # D_pth_path = 'netD_model_epoch_199.pth'

    parser = argparse.ArgumentParser(description='DeepRendering-server')
    parser.add_argument('--n_channel_input', type=int,
                        default=4, help='number of input channels')
    parser.add_argument('--n_channel_output', type=int,
                        default=4, help='number of output channels')
    parser.add_argument('--n_generator_filters', type=int,
                        default=64, help='number of initial generator filters')
    opt = parser.parse_args()

    loaded_model = torch.load(G_pth_path, map_location=torch.device('cpu'))
    netG = G(opt.n_channel_input * 4, opt.n_channel_output, opt.n_generator_filters)
    netG.load_state_dict(loaded_model['state_dict_G'])

    print(" starting service ...")

    app.run()
    
    # clean up after flask is stopped
    os.remove('albedo.png')
    os.remove('normal.png')
    os.remove('depth.png')
    os.remove('direct.png')
    os.remove('result.png')