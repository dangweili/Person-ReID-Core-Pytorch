import os
import sys
import h5py

from zipfile import ZipFile
from scipy.misc import imsave
from itertools import chain
import cPickle as pickle
import numpy as np
import random
import pdb

random.seed(0)
np.random.seed(0)

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def unzip_viper_data(zip_file, save_dir):
    print "Extracting viper zip file:\n"
    print zip_file
    make_dir(save_dir) 
    with ZipFile(zip_file) as f:
        f.extractall(save_dir)
    print "Extracting viper zip file done!\n"

def parse_image_name(img_name):
    pid = int(img_name.split('_')[0])
    return pid, 1, 1, 1, 1

def generate_dataset_description(save_dir):
    dataset = {}
    dataset['description'] = 'viper'
    dataset['root'] = './dataset/viper'
    dataset['image'] = []
    dataset['pid'] = []
    dataset['cam'] = []
    dataset['seq'] = []
    dataset['frame'] = []
    dataset['record'] = []
    import glob
    fs = glob.glob(os.path.join(save_dir, 'VIPeR/cam_a', '*.bmp'))
    for f in fs:
        basename = os.path.basename( f )
        pid, cam, seq, frame, record = parse_image_name( basename )
        dataset['image'].append(os.path.join('VIPeR/cam_a', basename))
        dataset['pid'].append(pid)
        dataset['cam'].append(1)
        dataset['seq'].append(1)
        dataset['frame'].append(1)
        dataset['record'].append(1)

    fs = glob.glob(os.path.join(save_dir, 'VIPeR/cam_b', '*.bmp'))
    for f in fs:
        basename = os.path.basename( f )
        pid, cam, seq, frame, record = parse_image_name( basename )
        dataset['image'].append(os.path.join('VIPeR/cam_b', basename))
        dataset['pid'].append(pid)
        dataset['cam'].append(2)
        dataset['seq'].append(1)
        dataset['frame'].append(1)
        dataset['record'].append(1)
    with open(os.path.join(save_dir, 'viper_dataset.pkl'), 'w+') as f:
        pickle.dump(dataset, f)

def create_trainvaltest_split(save_dir, split_file, val_cnt=100):
    pids = []
    import glob
    fs = glob.glob(os.path.join(save_dir, 'VIPeR/cam_a', '*.bmp'))
    for f in fs:
        basename = os.path.basename( f )
        pid = int(basename.split('_')[0])
        pids.append( pid )
    partition = dict()
    partition['trainval'] = []
    partition['train'] = []
    partition['val'] = []
    partition['test'] = []
    import copy
    for _ in range(100):
        tmp = copy.deepcopy( pids )
        random.shuffle( tmp )
        partition['trainval'].append(tmp[:316])
        partition['train'].append(tmp[:316-val_cnt])
        partition['val'].append(tmp[316-val_cnt:316])
        partition['test'].append(tmp[316:])
    with open(split_file, 'w+') as f:
        pickle.dump(partition, f)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="viper dataset")
    parser.add_argument(
        '--zip_file',
        type=str,
        default="./dataset/viper/VIPeR.v1.0.zip")
    parser.add_argument(
        '--save_dir',
        type=str,
        default="./dataset/viper/")
    parser.add_argument(
        '--traintest_split_file',
        type=str,
        default="./dataset/viper/viper_partition.pkl")

    args = parser.parse_args()
    zip_file = args.zip_file
    traintest_split_file = args.traintest_split_file
    save_dir = args.save_dir
    
    unzip_viper_data(zip_file, save_dir)
    generate_dataset_description(save_dir)
    create_trainvaltest_split(save_dir=save_dir, split_file=traintest_split_file, val_cnt=100)
