import time
import os
import shutil
from iterativeutil import *

def train_epoch():

    # record the train directory file.
    train_list = os.listdir(os.path.join(datasetdirT,"albedo"))
    train_list = [str(get_timestamp(x)) for x in train_list]

    print("training ... ")
    # run the training process.
    os.sys.path.append("..") 
    import train_backup

    # time.sleep(10)

    # after which, clean up the previous model.
    # os.remove(train_backup.opt.resume_G)
    # os.remove(train_backup.opt.resume_D)

    # clean up val dir after training.
    for buffer in ['albedo', 'direct', 'normal', 'depth', 'gt']:
        val_dir = os.path.join(datasetdirV,buffer)
        shutil.rmtree(val_dir)
        os.mkdir(val_dir)
    # move the train data to val.
    for buffer in ['albedo', 'direct', 'normal', 'depth', 'gt']:
        for i in range(len(train_list)):
            filename = buffer + "_" + train_list[i] + ".png"
            shutil.move(os.path.join(datasetdirT,buffer,filename), os.path.join(datasetdirV,buffer,filename))

if __name__ == "__main__":
    while(True):
        train_epoch()
        time.sleep(10)