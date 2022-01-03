## DeepIllumination

### How to run the code?
To run the project you will need:
 * python 3.5
 * pytorch (CUDA)
 * tensorboard

1. Please put your dataset under the directory ./dataset in a structure like below:
```
├───dataset
│   ├───JiaRan              # name of your dataset
│   │   ├───train
│   │   │   ├───albedo
│   │   │   ├───depth
│   │   │   ├───direct
│   │   │   ├───gt
│   │   │   └───normal
│   │   └───val
│   │       ├───albedo
│   │       ├───depth
│   │       ├───direct
│   │       ├───gt
│   │       └───normal
```

2. To train the model:
```
python train.py --dataset dataset/[name of your dataset]/ --n_epochs num of epochs
```
Check [train.py](train.py) for more options.

3. To test the model:
```
python test.py --dataset dataset/[name of your dataset]/ --model checkpoint/[name of your checkpoint]
```
Check [result/](result/) for the output.

### How to run blender plugin?

1. Download modified blender v2.79 from [blender-custom-nodes](https://github.com/bitsawer/blender-custom-nodes), open the example project `blender/CornellBoxDegraded.blend`. **Scene** is in EEVEE renderer for inferring only; **Scene.001** is in Cycles renderer for additional training.

> You need to move your trained model in `checkpoint/[name of your dataset]` directory
>  * `netD_model_epoch_199.pth`
>  * `netG_model_epoch_199.pth`
>
> to directory `ui/checkpoint` first, before starting the servers.

2. To open the infer server (necessary):
```
cd ui
python server-iterative.py
```

3. To open the trainer server:
```
cd ui
python trainer-iterative.py
```

More information about the plugin, refer to [blender/README.md](blender/README.md).