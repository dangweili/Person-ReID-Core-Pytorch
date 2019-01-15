import os
import sys
from zipfile import ZipFile
import pickle
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

def unzip_dukemtmcreid_data(zip_file, save_dir):
    print("Extracting dukemtmcreid zip file")
    print(zip_file)
    make_dir(save_dir)
    with ZipFile(zip_file) as f:
        f.extractall(save_dir)
    print("Extracting dukemtmcreid zip file done!")

def parse_image_name( img_name ):
    # pid, cam, seq, frame, record
    strs = img_name.split('.')[0].split('_')
    pid = int(strs[0])
    cam = int(strs[1][1])
    seq = 1
    frame = int(strs[2][1:])
    record = 1
    return pid, cam, seq, frame, record

def generate_dataset_description(save_dir):
    dataset = {}
    dataset['description'] = 'duke'
    dataset['root'] = './dataset/dukemtmcreid/'
    dataset['image'] = []
    dataset['pid'] = []
    dataset['cam'] = []
    dataset['seq'] = []
    dataset['frame'] = []
    dataset['record'] = []
    import glob
    # the training images
    img_path = os.path.join(save_dir, 'DukeMTMC-reID/bounding_box_train')
    fs = glob.glob(os.path.join(img_path, '*.jpg'))
    fs.sort()
    rpath = 'DukeMTMC-reID/bounding_box_train/'
    for f in fs:
        basename = os.path.basename(f)
        dataset['image'].append( rpath + basename )
        pid, cam, seq, frame, record = parse_image_name( basename )
        dataset['pid'].append(pid)
        dataset['cam'].append(cam)
        dataset['seq'].append(seq)
        dataset['frame'].append(frame)
        dataset['record'].append(record)
    # the test images
    dataset['image_g'] = []
    dataset['pid_g'] = []
    dataset['cam_g'] = []
    dataset['seq_g'] = []
    dataset['frame_g'] = []
    dataset['record_g'] = []
    img_path = os.path.join(save_dir, 'DukeMTMC-reID/bounding_box_test')
    fs = glob.glob(os.path.join(img_path, '*.jpg'))
    fs.sort()
    rpath = 'DukeMTMC-reID/bounding_box_test/'
    for f in fs:
        basename = os.path.basename(f)
        dataset['image_g'].append( rpath + basename)
        pid, cam, seq, frame, record = parse_image_name( basename )
        dataset['pid_g'].append(pid)
        dataset['cam_g'].append(cam)
        dataset['seq_g'].append(seq)
        dataset['frame_g'].append(frame)
        dataset['record_g'].append(record)
    # the fixed query set
    dataset['image_q'] = []
    dataset['pid_q'] = []
    dataset['cam_q'] = []
    dataset['seq_q'] = []
    dataset['frame_q'] = []
    dataset['record_q'] = []
    img_path = os.path.join(save_dir, 'DukeMTMC-reID/query')
    fs = glob.glob(os.path.join(img_path, '*.jpg'))
    fs.sort()
    rpath = 'DukeMTMC-reID/query/'
    for f in fs:
        basename = os.path.basename(f)
        dataset['image_q'].append( rpath + basename)
        pid, cam, seq, frame, record = parse_image_name( basename )
        dataset['pid_q'].append(pid)
        dataset['cam_q'].append(cam)
        dataset['seq_q'].append(seq)
        dataset['frame_q'].append(frame)
        dataset['record_q'].append(record)
    dataset['root'] = './dataset/dukemtmcreid'
    with open(os.path.join(save_dir, 'dukemtmcreid_dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

def create_trainvaltest_split(save_dir, traintest_split_file, val_cnt=100):
    # load training identity
    trainval_identity = {}
    img_path = os.path.join(save_dir, 'DukeMTMC-reID/bounding_box_train')
    import glob
    fs = glob.glob(os.path.join(img_path, '*.jpg'))
    for f in fs:
        basename = os.path.basename(f)
        pid, cam, seq, frame, record = parse_image_name( basename )
        trainval_identity[pid] = 1
    trainval_identity = list(trainval_identity.keys())
    random.shuffle(trainval_identity)
    train_identity = trainval_identity[:-val_cnt]
    val_identity = trainval_identity[-val_cnt:]
    
    test_identity = {}
    img_path = os.path.join(save_dir, 'DukeMTMC-reID/bounding_box_test')
    import glob
    fs = glob.glob(os.path.join(img_path, '*.jpg'))
    for f in fs:
        basename = os.path.basename(f)
        pid, cam, seq, frame, record = parse_image_name( basename )
        test_identity[pid] = 1
    test_identity = list(test_identity.keys())
    partition = dict()
    partition['trainval'] = [trainval_identity]
    partition['val'] = [val_identity]
    partition['train'] = [train_identity]
    partition['test'] = [test_identity]
    with open(traintest_split_file, 'wb+') as f:
        pickle.dump(partition, f)
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="dukemtmc reid dataset")
    parser.add_argument(
        '--zip_file',
        type=str,
        default="./dataset/dukemtmcreid/DukeMTMC-reID.zip")
    parser.add_argument(
        '--save_dir',
        type=str,
        default="./dataset/dukemtmcreid/")
    parser.add_argument(
        '--traintest_split_file',
        type=str,
        default="./dataset/dukemtmcreid/dukemtmcreid_partition.pkl")
    args = parser.parse_args()
    zip_file = args.zip_file
    save_dir = args.save_dir
    traintest_split_file = args.traintest_split_file
    
    unzip_dukemtmcreid_data(zip_file, save_dir)
    generate_dataset_description(save_dir)
    create_trainvaltest_split(save_dir, traintest_split_file, val_cnt=100)
