import os
import sys
import h5py

from zipfile import ZipFile
from scipy.misc import imsave
from scipy.io import loadmat
from itertools import chain
import pickle
import numpy as np
import random
import pdb

random.seed(0)
np.random.seed(0)

global_list_det = []
global_list_lab = []

def make_dir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def unzip_cuhk03_data(zip_file, save_dir):
    print("Extracting cuhk03 zip file:\n")
    print(zip_file)
    make_dir(save_dir) 
    with ZipFile(zip_file) as f:
        f.extractall(save_dir)
    print("Extracting cuhk03 zip file done!\n")

def parse_image_name(img_name):
    if len(img_name) != 40:
        return None,None,None,None,None
    pid = int(img_name[:8])
    cam = int(img_name[9:13])
    seq = int(img_name[14:18])
    frame = int(img_name[19:27])
    record = int(img_name[28:36])
    return pid, cam, seq, frame, record

def create_unified_data(save_dir):
    mat_file = os.path.join(save_dir, 'cuhk03_release/cuhk-03.mat')
    print("Creating unified data: ID8_CAM4_SEQ4_FRAME8_RECORD8")
    img_name_tmp = '{:08d}_{:04d}_{:04d}_{:08d}_{:08d}.png'
    def deref(mat, ref):
        return mat[ref][:].T
    def dump(mat, ref, pid, cam, seq, frame, record, im_dir):
        im = deref(mat, ref)
        if im.size == 0 or im.ndim <2: 
            return ''
        fname = img_name_tmp.format(pid, cam, seq, frame, record)
        imsave(os.path.join(im_dir, fname), im)
        return fname
    # parsing the mat file
    pid = 1 
    mat = h5py.File(mat_file, 'r')
    labeled_im_dir = os.path.join(save_dir, 'labeled')
    detected_im_dir = os.path.join(save_dir, 'detected')
    make_dir(labeled_im_dir)
    make_dir(detected_im_dir)
    for labeled, detected in zip(mat['labeled'][0], mat['detected'][0]):
        labeled, detected = deref(mat, labeled), deref(mat, detected)
        # loop images for a camera pair
        for i in range(labeled.shape[0]):
            for j in range(5):
                fname = dump(mat, labeled[i, j], pid, 1, j+1, 1, 1, labeled_im_dir)
                if fname != '':
                    global_list_lab.append(os.path.join('labeled', fname))
                fname = dump(mat, detected[i, j], pid, 1, j+1, 1, 1, detected_im_dir)
                if fname != '':
                    global_list_det.append(os.path.join('detected', fname))
            for j in range(5):
                fname = dump(mat, labeled[i, j+5], pid, 2, j+1, 1, 1, labeled_im_dir)
                if fname != '':
                    global_list_lab.append(os.path.join('labeled', fname))
                fname = dump(mat, detected[i, j+5], pid, 2, j+1, 1, 1, detected_im_dir)
                if fname != '':
                    global_list_det.append(os.path.join('detected', fname))
            if pid % 100 == 0:
                print('Saving images {}/{}'.format(pid, 1467))
            pid = pid + 1
def generate_dataset_description(save_dir):
    labeled_im_dir = os.path.join(save_dir, 'labeled')
    detected_im_dir = os.path.join(save_dir, 'detected')
    # generate the dataset image list
    dataset = {}
    dataset['description'] = 'cuhk03'
    dataset['root'] = './dataset/cuhk03/'
    dataset['image'] = []
    dataset['pid'] = []
    dataset['cam'] = []
    dataset['seq'] = []
    dataset['frame'] = []
    dataset['record'] = []
    import glob
    fs = glob.glob(os.path.join(labeled_im_dir, '*.png'))
    for f in fs:
        dataset['image'].append('labeled/%s'%(os.path.basename(f)))
        pid, cam, seq, frame, record = parse_image_name(os.path.basename(f))
        dataset['pid'].append(pid)
        dataset['cam'].append(cam)
        dataset['seq'].append(seq)
        dataset['frame'].append(frame)
        dataset['record'].append(record)
    dataset['root'] = './dataset/cuhk03'
    # for new protocol, using fixed query/gallery partition
    dataset['image_q'] = []
    dataset['pid_q'] = []
    dataset['cam_q'] = []
    dataset['seq_q'] = []
    dataset['frame_q'] = []
    dataset['record_q'] = []
    mat_file = './dataset/cuhk03/cuhk03_new_protocol_config_labeled.mat'
    mat = loadmat(open(mat_file, 'r'))
    for i in range(mat['query_idx'].shape[0]):
        idx = mat['query_idx'][i, 0]-1
        pid, cam, seq, frame, record = parse_image_name(os.path.basename(global_list_lab[idx]))
        dataset['image_q'].append('labeled/%s'%(os.path.basename(global_list_lab[idx])))
        dataset['pid_q'].append(pid)
        dataset['cam_q'].append(cam)
        dataset['seq_q'].append(seq)
        dataset['frame_q'].append(frame)
        dataset['record_q'].append(record)
    dataset['image_g'] = []
    dataset['pid_g'] = []
    dataset['cam_g'] = []
    dataset['seq_g'] = []
    dataset['frame_g'] = []
    dataset['record_g'] = []
    for i in range(mat['gallery_idx'].shape[0]):
        idx = mat['gallery_idx'][i, 0]-1
        pid, cam, seq, frame, record = parse_image_name(os.path.basename(global_list_lab[idx]))
        dataset['image_g'].append('labeled/%s'%(os.path.basename(global_list_lab[idx])))
        dataset['pid_g'].append(pid)
        dataset['cam_g'].append(cam)
        dataset['seq_g'].append(seq)
        dataset['frame_g'].append(frame)
        dataset['record_g'].append(record)
    with open(os.path.join(save_dir, 'cuhk03_labeled_dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)
    dataset = {}
    dataset['description'] = 'cuhk03'
    dataset['image'] = []
    dataset['pid'] = []
    dataset['cam'] = []
    dataset['seq'] = []
    dataset['frame'] = []
    dataset['record'] = []
    import glob
    fs = glob.glob(os.path.join(detected_im_dir, '*.png'))
    for f in fs:
        dataset['image'].append('detected/%s'%(os.path.basename(f)))
        pid, cam, seq, frame, record = parse_image_name(os.path.basename(f))
        dataset['pid'].append(pid)
        dataset['cam'].append(cam)
        dataset['seq'].append(seq)
        dataset['frame'].append(frame)
        dataset['record'].append(record)
    dataset['root'] = './dataset/cuhk03'
    mat_file = './dataset/cuhk03/cuhk03_new_protocol_config_detected.mat'
    mat = loadmat(open(mat_file, 'r'))
    dataset['image_q'] = []
    dataset['pid_q'] = []
    dataset['cam_q'] = []
    dataset['seq_q'] = []
    dataset['frame_q'] = []
    dataset['record_q'] = []
    for i in range(mat['query_idx'].shape[0]):
        idx = mat['query_idx'][i, 0] - 1
        pid, cam, seq, frame, record = parse_image_name(os.path.basename(global_list_det[idx]))
        dataset['image_q'].append('detected/%s'%(os.path.basename(global_list_det[idx])))
        dataset['pid_q'].append(pid)
        dataset['cam_q'].append(cam)
        dataset['seq_q'].append(seq)
        dataset['frame_q'].append(frame)
        dataset['record_q'].append(record)
    dataset['image_g'] = []
    dataset['pid_g'] = []
    dataset['cam_g'] = []
    dataset['seq_g'] = []
    dataset['frame_g'] = []
    dataset['record_g'] = []
    for i in range(mat['gallery_idx'].shape[0]):
        idx = mat['gallery_idx'][i, 0] - 1
        pid, cam, seq, frame, record = parse_image_name(os.path.basename(global_list_det[idx]))
        dataset['image_g'].append('detected/%s'%(os.path.basename(global_list_det[idx])))
        dataset['pid_g'].append(pid)
        dataset['cam_g'].append(cam)
        dataset['seq_g'].append(seq)
        dataset['frame_g'].append(frame)
        dataset['record_g'].append(record)
    with open(os.path.join(save_dir, 'cuhk03_detected_dataset.pkl'), 'wb+') as f:
        pickle.dump(dataset, f)

def create_trainvaltest_split_old(save_dir, split_file, val_cnt=100):
    mat_file = os.path.join(save_dir, 'cuhk03_release/cuhk-03.mat')
    mat = h5py.File(mat_file, 'r')
    # two types of parttion using the identity
    IDs = []
    for i in range(mat['labeled'][0].shape[0]):
        IDs.append(mat[mat['labeled'][0][i]].shape[1])
    startIDs = [0]
    for i in range(len(IDs)):
        startIDs.append(IDs[i] + startIDs[i])
    if startIDs[-1] != 1467:
        print('person identities are not consistent')
    # 20 times
    partition = dict()
    partition['trainval'] = []
    partition['train'] = []
    partition['val'] = []
    partition['test'] = []
    for i in range(mat['testsets'][0].shape[0]):
        test_identity = []
        testIDs = mat[mat['testsets'][0][i]][:].T # 100*2
        for j in range(testIDs.shape[0]):
            test_identity.append(startIDs[int(testIDs[j, 0])-1] + int(testIDs[j, 1]))
        tmp_identitys = range(1, 1468)
        random.shuffle(tmp_identitys)
        trainval_identity = [pid for pid in tmp_identitys if pid not in test_identity]
        train_identity = trainval_identity[:-val_cnt] 
        val_identity = trainval_identity[-val_cnt:]
        partition['trainval'].append(trainval_identity)
        partition['train'].append(train_identity)
        partition['val'].append(val_identity)
        partition['test'].append(test_identity)
    with open(split_file, 'wb+') as f:
        pickle.dump(partition, f)

#  
def create_trainvaltest_split_new(save_dir, label, split_file, val_cnt=100):
    ''' the new protocol from zhongzhun '''
    # labeled part
    if label == 'labeled':
        mat_file = './dataset/cuhk03/cuhk03_new_protocol_config_labeled.mat'
    elif label == 'detected':
        mat_file = './dataset/cuhk03/cuhk03_new_protocol_config_detected.mat'
    else:
        print('%s not supported'%(label))
        raise ValueError

    mat = loadmat(open(mat_file, 'r'))
    partition = {}
    partition['trainval'] = []
    partition['train'] = []
    partition['val'] = []
    partition['test'] = []
    # load the predfined train/test split
    trainval_identity = dict() 
    for i in range(mat['train_idx'].shape[0]):
        idx = mat['train_idx'][i, 0] - 1
        if label == 'labeled':
            pid, cam, seq, frame, record = parse_image_name(os.path.basename(global_list_lab[idx]))
        else:
            pid, cam, seq, frame, record = parse_image_name(os.path.basename(global_list_det[idx]))
        trainval_identity[pid] = 1
    trainval_identity = trainval_identity.keys()
    random.shuffle(trainval_identity)
    train_identity = trainval_identity[:-val_cnt]
    val_identity = trainval_identity[-val_cnt:]
    tmp_identitys = range(1, 1468)
    test_identity = [pid for pid in tmp_identitys if pid not in trainval_identity]
    partition['trainval'].append(trainval_identity)
    partition['train'].append(train_identity)
    partition['val'].append(val_identity)
    partition['test'].append(test_identity)
    with open(split_file, 'wb+') as f:
        pickle.dump(partition, f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="cuhk03 dataset")
    parser.add_argument(
        '--zip_file',
        type=str,
        default="./dataset/cuhk03/cuhk03-release.zip")
    parser.add_argument(
        '--save_dir',
        type=str,
        default="./dataset/cuhk03/")
    parser.add_argument(
        '--traintest_split_file',
        type=str,
        default="./dataset/cuhk03/cuhk03_partition.pkl")
    args = parser.parse_args()
    zip_file = args.zip_file
    traintest_split_file = args.traintest_split_file
    save_dir = args.save_dir
    
    unzip_cuhk03_data(zip_file, save_dir)
    create_unified_data(save_dir)
    generate_dataset_description(save_dir)
    traintest_split_file = "./dataset/cuhk03/cuhk03_partition_old.pkl"
    create_trainvaltest_split_old(save_dir=save_dir, split_file=traintest_split_file, val_cnt=100)
    
    label = 'labeled'
    traintest_split_file = "./dataset/cuhk03/cuhk03_partition_new_%s.pkl"%(label)
    create_trainvaltest_split_new(save_dir=save_dir, label=label, split_file=traintest_split_file, val_cnt=100)
    
    label = 'detected'
    traintest_split_file = "./dataset/cuhk03/cuhk03_partition_new_%s.pkl"%(label)
    create_trainvaltest_split_new(save_dir=save_dir, label=label, split_file=traintest_split_file, val_cnt=100)
