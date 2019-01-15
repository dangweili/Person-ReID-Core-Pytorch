#!/usr/bin/env python

import os
import sys
import numpy as np
import pickle
import random
from zipfile import ZipFile
import re
from scipy.io import loadmat
import pdb

np.random.seed(0)
random.seed(0)
def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def unzip_mars_data(zip_file, save_dir):
    print("Extracting mars zip files")
    print(zip_file)
    make_dir(save_dir)
    with ZipFile(zip_file) as f:
        f.extractall(save_dir)
    print("Extracting mars zip file done!")

def parse_image_name( img_name ):
    # pid, cam, seq, frame, record
    pid_str = re.sub('^[0]*', '', img_name[:4])
    if pid_str == '':
        pid = 0
    else:
        pid = int(pid_str)
    cam = int(img_name[5])
    seq = int(img_name[7:11])
    frame = int(img_name[12:15])
    record = 1
    return pid, cam, seq, frame, record

def generate_data_description(save_dir):
    dataset = {}
    dataset['description'] = 'mars'
    dataset['root'] = './dataset/mars/'
    dataset['image'] = []
    dataset['pid'] = []
    dataset['cam'] = []
    dataset['seq'] = []
    dataset['frame'] = []
    dataset['record'] = []
    import glob
    # the training images
    rpath = 'bbox_train/'
    img_path = os.path.join(save_dir, rpath)
    dirs = os.listdir(img_path)
    dirs.sort()
    for f in dirs:
        imgs = glob.glob(os.path.join(img_path, f, '*.jpg'))
        imgs.sort()
        for img in imgs:
            basename = os.path.basename(img)
            pid, cam, seq, frame, record = parse_image_name( basename )
            dataset['image'].append(os.path.join(rpath, f, basename))
            dataset['pid'].append(pid)
            dataset['cam'].append(cam)
            dataset['seq'].append(seq)
            dataset['frame'].append(frame)
            dataset['record'].append(record)
    # the test images
    rpath = 'bbox_test/'
    img_path = os.path.join(save_dir, rpath)
    dirs = os.listdir(img_path)
    dirs.sort()
    for f in dirs:
        imgs = glob.glob(os.path.join(img_path, f, '*.jpg'))
        imgs.sort()
        for img in imgs:
            basename = os.path.basename(img)
            pid, cam, seq, frame, record = parse_image_name( basename )
            dataset['image'].append(os.path.join(rpath, f, basename))
            dataset['pid'].append(pid)
            dataset['cam'].append(cam)
            dataset['seq'].append(seq)
            dataset['frame'].append(frame)
            dataset['record'].append(record)
    # using fixed query/gallery partition for video-based person re-identification
    data = loadmat(os.path.join(save_dir, 'MARS-evaluation/info/tracks_test_info.mat'))
    track_test = data['track_test_info']
    data = loadmat(os.path.join(save_dir, 'MARS-evaluation/info/query_IDX.mat'))
    query_IDX = data['query_IDX']
    test_name = []
    fid = open(os.path.join(save_dir, 'MARS-evaluation/info/test_name.txt'))
    for line in fid.readlines():
        test_name.append(line.strip())
    # using fixed query, including the id, cam, seq number
    dataset['track_pid_q'] = []
    dataset['track_cam_q'] = []
    dataset['track_seq_q'] = []
    for i in range(query_IDX.shape[1]):
        basename = test_name[track_test[query_IDX[0,i]-1, 0]-1]
        pid, cam, seq, frame, record = parse_image_name( basename )
        dataset['track_pid_q'].append(pid)
        dataset['track_cam_q'].append(cam)
        dataset['track_seq_q'].append(seq)
    # using fixed gallery, including the id, cam, seq number
    dataset['track_pid_g'] = []
    dataset['track_cam_g'] = []
    dataset['track_seq_g'] = []
    for i in range(track_test.shape[0]):
        basename = test_name[track_test[i, 0]-1]
        pid, cam, seq, frame, record = parse_image_name( basename )
        dataset['track_pid_g'].append(pid)
        dataset['track_cam_g'].append(cam)
        dataset['track_seq_g'].append(seq)
    
    dataset['root'] = './dataset/mars'
    with open(os.path.join(save_dir, 'mars_dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

def create_trainvaltest_split(save_dir, split_file, val_cnt=100):
    # load training identity
    trainval_identity = {}
    rpath = 'bbox_train'
    dirs = os.listdir(os.path.join(save_dir, rpath))
    for f in dirs:
        pid = int(f)
        trainval_identity[pid] = 1
    trainval_identity = list(trainval_identity.keys())
    random.shuffle(trainval_identity)
    train_identity = trainval_identity[:-val_cnt]
    val_identity = trainval_identity[-val_cnt:]
    test_identity = {}
    rpath = 'bbox_test'
    dirs = os.listdir(os.path.join(save_dir, rpath))
    for f in dirs:
        pid_str = re.sub(r'^[0]*', '', f)
        if pid_str == '':
            pid = 0
        else:
            pid = int(pid_str)
        test_identity[pid] = 1
    test_identity = list(test_identity.keys())
    partition = dict()
    partition['trainval'] = [trainval_identity]
    partition['val'] = [val_identity]
    partition['train'] = [train_identity]
    partition['test'] = [test_identity]
    with open(split_file, 'wb+') as f:
        pickle.dump(partition, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="mars dataset")
    parser.add_argument(
        '--zip_file',
        type=str,
        default="./dataset/mars/MARS-v160809/bbox_train.zip")
    parser.add_argument(
        '--save_dir',
        type=str,
        default="./dataset/mars/")
    parser.add_argument(
        '--traintest_split_file',
        type=str,
        default="./dataset/mars/mars_partition.pkl")
    args = parser.parse_args()
    zip_file = args.zip_file
    save_dir = args.save_dir
    traintest_split_file = args.traintest_split_file

    unzip_mars_data(zip_file, save_dir)
    zip_file = zip_file.replace('bbox_train.zip', 'bbox_test.zip')
    unzip_mars_data(zip_file, save_dir)
    generate_data_description(save_dir)
    create_trainvaltest_split(save_dir, traintest_split_file, val_cnt=100)
    pass
