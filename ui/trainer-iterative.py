import time
import os
import shutil
from iterativeutil import *

def train_epoch():

    # record the train directory file.
    train_list = os.listdir(os.path.join(datasetdirT,"albedo"))
    train_list = [x.split(".")[0] for x in train_list]
    if len(train_list) == 0:
        return 

    print("training ... ")
    # run the training process.
    os.sys.path.append("..") 
    import train_backup

    # time.sleep(10)
    
    G_filelist = os.listdir(checkpointdir)
    G_filelist = [x for x in G_filelist if x.startswith('netG')]
    G_filelist.sort()
    if len(G_filelist)>0:
        lastEpoch = int(G_filelist[-1].split("_")[3].split(".")[0])
    else:
        lastEpoch = 0

    for i in range(train_backup.opt.n_epoch):
        train_backup.train(lastEpoch+i)
    train_backup.save_checkpoint(lastEpoch+1)

    # after which, clean up the previous model.
    if train_backup.opt.resume_G:
        os.remove(train_backup.opt.resume_G)
        os.remove(train_backup.opt.resume_D)

    # clean up val dir after training.
    for buffer in ['albedo', 'direct', 'normal', 'depth', 'gt']:
        val_dir = os.path.join(datasetdirV,buffer)
        shutil.rmtree(val_dir)
        os.mkdir(val_dir)
    # move the train data to val.
    for buffer in ['albedo', 'direct', 'normal', 'depth', 'gt']:
        for i in range(len(train_list)):
            filename = train_list[i] + ".png"
            shutil.move(os.path.join(datasetdirT,buffer,filename), os.path.join(datasetdirV,buffer,filename))
    
    time.sleep(2)

if __name__ == "__main__":
    if not os.path.isdir(checkpointdir):
        os.mkdir(checkpointdir)

    while(True):
        train_epoch()
        time.sleep(10)