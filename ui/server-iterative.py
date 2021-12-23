from time import time
import torch
import argparse
from flask import Flask, request, render_template, Response
import shutil
import os
os.sys.path.append("..") 
from model_backup import G,D
from util import load_image, save_image

app = Flask(__name__)

datasetdir = 'dataset'
checkpointdir = 'checkpoint'

prev_G_path = 'default.pth'

N_CHANNEL_INPUT = 4
N_CHANNEL_OUTPUT = 4
N_GENERATOR_FILTERS = 64

def load_model():
    global netG

    checkpointlist = os.listdir(checkpointdir)
    checkpointlist.sort()
    prev_G_path = checkpointlist[-1]
    
    print("loading model " + prev_G_path + " ...")
    loaded_model = torch.load(os.path.join(checkpointdir,prev_G_path), map_location=torch.device('cpu'))
    netG = G(N_CHANNEL_INPUT * 4, N_CHANNEL_OUTPUT, N_GENERATOR_FILTERS)
    netG.load_state_dict(loaded_model['state_dict_G'])


def infer():
    """
    Infer the result by plugin the inputs from save file.
    The output will be saved in result.png.
    """

    if not os.path.exists(prev_G_path):
        load_model()

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

    global netG
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

    albedo_file = request.files.get('albedo')
    normal_file = request.files.get('normal')
    depth_file = request.files.get('depth')
    direct_file = request.files.get('direct')

    albedo_filename = albedo_file.filename
    normal_filename = normal_file.filename
    depth_filename = depth_file.filename
    direct_filename = direct_file.filename

    # a copy for training
    albedo_file.save(albedo_filename)
    normal_file.save(normal_filename)
    depth_file.save(depth_filename)
    direct_file.save(direct_filename)

    # a copy for response
    # cannot use the same object for the defect.
    # TODO: use bitstream to avoid overhead.
    shutil.copyfile(albedo_filename, 'albedo.png')
    shutil.copyfile(normal_filename, 'normal.png')
    shutil.copyfile(depth_filename, 'depth.png')
    shutil.copyfile(direct_filename, 'direct.png')

    infer()

    return index('result')

@app.route('/gtupload', methods=['post'])
def gttrain():

    def get_timestamp(filename):
        # make sure it contains the _
        return int(filename.split('_')[1].split('.')[0])

    gt_file = request.files.get('gt')
    gt_filename = gt_file.filename
    gt_timestamp = get_timestamp(gt_filename)

    # only start training after the gt is received.
    # now, there will be some training samples suffixed by time.
    for item in os.listdir('.'):
        if len(item.split('_')) > 1:
            # format in albedo_[timestamp].png
            timestamp = get_timestamp(item)
            timedelta = gt_timestamp - timestamp
            if timedelta < 10 and timedelta >= 0:
                # use this batch for training.
                for buffer in ["albedo", "direct", "normal", "depth"]:
                    shutil.move(buffer + '_' + str(timestamp) + '.png', os.path.join(datasetdir, buffer))
                gt_file.save(os.path.join(datasetdir, "gt", "gt_" + str(timestamp) + ".png"))
                break   # no more move is needed.

    return Response()   # gt is received.

@app.route('/')
def root():
    return render_template("index.html")

if __name__ == "__main__":
    print(" starting service ...")

    if not os.path.isdir(datasetdir):
        os.mkdir(datasetdir)
        os.mkdir(os.path.join(datasetdir, 'albedo'))
        os.mkdir(os.path.join(datasetdir, 'depth'))
        os.mkdir(os.path.join(datasetdir, 'normal'))
        os.mkdir(os.path.join(datasetdir, 'direct'))
        os.mkdir(os.path.join(datasetdir, 'gt'))

    if not os.path.isdir(checkpointdir):
        os.mkdir(checkpointdir)

    app.run()
    
    # clean up after flask is stopped
    for item in os.listdir('.'):
        if item.endswith(".png"):
            os.remove(item)