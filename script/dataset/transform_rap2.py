import os
import sys
import cPickle as pickle
import numpy as np
import random
import pdb
from scipy.io import loadmat

random.seed(0)
np.random.seed(0)

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

def parse_image_name(img_name):
    cam = int(img_name[3:5])
    year = img_name[6:10]
    month = img_name[11:13]
    day = img_name[14:16]
    date = int(year + month + day)
    seq = int(img_name[17:31])
    frame = int(img_name[17:].split('.')[0].split('-')[3][5:])
    record = 1
    return cam, date, seq, frame, record
    
def generate_dataset_description(save_dir):
    dataset = {}
    dataset['description'] = 'rap2'
    dataset['root'] = './dataset/rap2/RAP_dataset/'
    # for person re-identification
    dataset['day'] = []
    dataset['image'] = []
    dataset['pid'] = []
    dataset['cam'] = []
    dataset['seq'] = []
    dataset['frame'] = []
    dataset['record'] = []
    # for pedestrian attriubte
    dataset['att'] = []
    dataset['att_name'] = []
    dataset['selected_attribute'] = []
    
    # load mat
    data = loadmat(open('./dataset/rap2/RAP_annotation/RAP_annotation.mat', 'r'))
    
    import pdb
    pdb.set_trace()
    dataset['selected_attribute'] = (data['RAP_annotation'][0][0][3][0,:]-1).tolist()
    
    for idx in range(152):
        dataset['att_name'].append(data['RAP_annotation'][0][0][2][idx][0][0])

    for idx in range(84928):
        # name and attribute
        name = data['RAP_annotation'][0][0][0][idx][0][0]
        dataset['image'].append(name)
        dataset['att'].append(data['RAP_annotation'][0][0][1][idx, :].tolist())
        # person id and other necessary information
        dataset['pid'].append(data['RAP_annotation'][0][0][5][idx, 0])
        cam, day, seq, frame, record = parse_image_name(name)
        dataset['cam'].append(cam)
        dataset['day'].append(day)
        dataset['seq'].append(seq)
        dataset['frame'].append(frame)
        dataset['record'].append(record)
    
    # create fixed query and gallery images
    dataset['image_q'] = []
    dataset['pid_q'] = []
    dataset['cam_q'] = []
    dataset['day_q'] = []
    dataset['seq_q'] = []
    dataset['frame_q'] = []
    dataset['record_q'] = []
    query_file = './dataset/rap2/rap2_test_image_name.txt'
    if not os.path.exists(query_file):
        raise ValueError
    fid = open(query_file)
    query_image_names = []
    for line in fid.readlines():
        name = line.strip()
        query_image_names.append(name)
    fid.close()
    query_index = []
    for name in query_image_names:
        pos = dataset['image'].index(name)
        dataset['image_q'].append(name)
        dataset['pid_q'].append(dataset['pid'][pos])
        dataset['day_q'].append(dataset['day'][pos])
        dataset['cam_q'].append(dataset['cam'][pos])
        dataset['seq_q'].append(dataset['seq'][pos])
        dataset['frame_q'].append(dataset['frame'][pos])
        dataset['record_q'].append(dataset['record'][pos])
    # create gallery list
    dataset['image_g'] = []
    dataset['pid_g'] = []
    dataset['cam_g'] = []
    dataset['day_g'] = []
    dataset['seq_g'] = []
    dataset['frame_g'] = []
    dataset['record_g'] = []
    test_pids = data['RAP_annotation'][0][0][6][0][0][1][0,:].tolist()
    test_pids.append(-1)
    for pos, pid in enumerate(dataset['pid']):
        if pid not in test_pids:
            continue
        dataset['image_g'].append(dataset['image'][pos])
        dataset['pid_g'].append(dataset['pid'][pos])
        dataset['day_g'].append(dataset['day'][pos])
        dataset['cam_g'].append(dataset['cam'][pos])
        dataset['seq_g'].append(dataset['seq'][pos])
        dataset['frame_g'].append(dataset['frame'][pos])
        dataset['record_g'].append(dataset['record'][pos])
    # save the dataset
    with open(os.path.join(save_dir, 'rap2reid_dataset.pkl'), 'w+') as f:
        pickle.dump(dataset, f)

def create_trainvaltest_split(save_dir, val_cnt=100):
    """
    create a dataset split file, which consists of index of the train/val/test splits
    """
    partition = dict()
    partition['trainval'] = []
    partition['train'] = []
    partition['val'] = []
    partition['test'] = []
    # load mat
    data = loadmat(open('./dataset/rap2/RAP_annotation/RAP_annotation.mat', 'r'))

    trainval = data['RAP_annotation'][0][0][6][0][0][0][0,:].tolist()
    val = random.sample(trainval, val_cnt)
    train = [x for x in trainval if x not in val]
    test = data['RAP_annotation'][0][0][6][0][0][1][0,:].tolist()
    test.append(-1)

    partition['trainval'].append(trainval)
    partition['train'].append(train)
    partition['val'].append(val)
    partition['test'].append(test)
    with open(os.path.join(save_dir, 'rap2reid_partition.pkl'), 'w+') as f:
        pickle.dump(partition, f)
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="rap2 reid dataset")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./dataset/rap2/")
    args = parser.parse_args()
    save_dir = args.save_dir

    generate_dataset_description(save_dir)
    create_trainvaltest_split(save_dir, val_cnt=100)
    print "Hello World"
