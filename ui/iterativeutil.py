
datasetdirT = 'dataset/train'
datasetdirV = 'dataset/val'
checkpointdir = 'checkpoint'

def get_timestamp(filename):
    # make sure it contains the _
    return int(filename.split('_')[1].split('.')[0])

def get_buffername(buffer, timestamp):
    return buffer + '_' + str(timestamp) + '.png'