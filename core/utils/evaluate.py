import os
import torch
from torch.autograd import Variable
import numpy as np
import copy


# Testing
def reid_evaluate(feat_func, dataset, **kwargs):
    """
    Evaluate the person re-identification results
    Args:
        model: the re-id model
        dataset: the unified dataset interface, such as ReIDTestDataset
        eval_type: "ss" single shot, such as cuhk and viper
                   "ms" mutiple shot, such as cuhk
                   "sq" single query, such as market, cuhk, duke, mars
                   "mq" mutiple query, such as market, mars
        video: whether the sequence-based or image-based re-identification
    Return:
        result: a dictionary that record the results of different eval_types
        result['ss']['mAP']
        result['ss']['CMC'], CMC is a 1*G array
    """
    if kwargs.has_key('eval_video') and kwargs['eval_video']:
        return reid_evaluate_sequence(feat_func, dataset, **kwargs)
    else:
        return reid_evaluate_image(feat_func, dataset, **kwargs)

def extract_feat(feat_func, dataset, **kwargs):
    """
    extract feature for images
    """
    test_loader = torch.utils.data.DataLoader(
        dataset = dataset, batch_size = 32,
        num_workers = 2, pin_memory = True)
    # extract feature for all the images of test/val identities
    N = len(dataset.image)
    start = 0
    for ep, imgs in enumerate(test_loader):
        imgs_var = Variable(imgs, volatile=True).cuda()
        feat_tmp = feat_func( imgs_var )
        batch_size = feat_tmp.shape[0]
        if ep == 0:
            feat = np.zeros((N, feat_tmp.size/batch_size))
        feat[start:start+batch_size, :] = feat_tmp.reshape((batch_size, -1))
        start += batch_size
    
    if kwargs.has_key('feat_only') and kwargs['feat_only']:
        return feat
    
    pid = copy.deepcopy( dataset.pid )
    cam = copy.deepcopy( dataset.cam )
    seq = copy.deepcopy( dataset.seq )
    frame = copy.deepcopy( dataset.frame )
    record = copy.deepcopy( dataset.record )
    return feat, pid, cam, seq, frame, record

def normalize(nparray, order=2, axis=0):
    """ Normalize a N-D numpy array along the specified axis. """
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray/norm
    # return nparray/(norm + np.finfo(np.float32).eps)

def compute_dist(array1, array2, dist_type='euclidean_normL2', A=None, verbose=False):
    """Compute the manhattan, euclidean, cosine distance of all pairs
    Args:
        array1: numpy array with shape [m1, D]
        array2: numpy array with shape [m2, D]
        A: mapping matrix with shape [D, d]
        type: one of ['cosine', 'euclidean', 'mahalanobis']
        Mah: (x_i - x_j) * M * (x_i - x_j).T
    Returns:
        numpy array with shape [m1, m2]
    """
    assert dist_type in ['cosine', 'euclidean', 'mahalanobis', 'euclidean_normL2']
    assert len(array1.shape) == 2 and len(array2.shape) == 2
    assert array1.shape[1] == array2.shape[1]
    if verbose: 
        D = array1.shape[1]
        M1 = array1.shape[0]
        M2  = array2.shape[0]
        print('compute %s distance between matrix [%d, %d] and [%d, %d]' \
            %(dist_type, M1, D, M2, D))
    # for cosine similarity
    if dist_type == 'cosine':
        # we use negative cosine similarity as distance
        array1 = normalize(array1, order=2, axis=1)
        array2 = normalize(array2, order=2, axis=1)
        dist = -1 * np.matmul(array1, array2.T)
        return dist
    # for euclidean distance
    if dist_type == 'euclidean':
        # shape [m1, 1]
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        squared_dist = -2 * np.matmul(array1, array2.T) + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)
        return dist
    # for euclidean_normL2 distance
    if dist_type == 'euclidean_normL2':
        norm_array1 = normalize(array1, order=2, axis=1)
        norm_array2 = normalize(array2, order=2, axis=1)
        # shape [m1, 1]
        square1 = np.sum(np.square(norm_array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(norm_array2), axis=1)[np.newaxis, ...]
        squared_dist = -2 * np.matmul(norm_array1, norm_array2.T) + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)
        return dist
    # for manhattan distance
    if dist_type == 'mahalanobis':
        assert A is not None
        # dimension reduction
        v = np.sum(np.matmul(array1, A) * array1, axis=1)[..., np.newaxis]
        tmp = np.matmul(array2, A)
        u = np.sum(tmp * array2, axis=1)[np.newaxis, ...]
        squared_dist = -2 * np.matmul(tmp, array1.T) + v + u 
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)
        return dist

def feature_pooling(feat, **kwargs):
    """ pool the feature into a single vector
    Input:
        feat: N*D
    Output:
        feat: 1*D
    In the future, we will support other operation, such as 
    local feature pooling and normalization
    """
    assert type(feat) == np.ndarray
    if kwargs.has_key('feat_pool_type'):
        operation = kwargs['feat_pool_type']
    else:
        operation = 'average'

    if operation == 'average':
        return np.mean(feat, axis=0, keepdims=True)
    elif operation == 'max':
        return np.max(feat, axis=0, keepdims=True)
    else:
        print('The pooling operation should be in %s'%('average, max'))
        raise ValueError

def compute_score(dist_mat, query_pid, query_cam, gallery_pid, gallery_cam, seperate_cam=False):
    """
    Input:
        dist_mat, distance matrix with shape [M, N]
        query_pid, ndarray with shape [1, N]
        gallery_pid, ndarray with shape [1, M] 
        # pid = -1 are distractors
    Return:
        mAP, CMC
    """
    index = np.argsort(dist_mat, axis=1) # with shape [M,N]
    cmcs = np.zeros(dist_mat.shape)
    aps = np.zeros(dist_mat.shape[0])
    for i in range(dist_mat.shape[0]):
        # calc ap and cmc
        aps[i], cmcs[i, :] = compute_ap_cmc(query_pid[0, i], query_cam[0, i], \
            gallery_pid[:, index[i, :]], gallery_cam[:, index[i, :]], seperate_cam)
    return np.mean(aps), np.mean(cmcs, axis=0, keepdims=True)

def compute_ap_cmc(query_pid, query_cam, gallery_pids, gallery_cams, seperate_cam=False):
    """
    Input: 
        query_pid, a scalar
        query_cam, a scalar
        gallery_pids, a 1xN ndarray
        gallery_cams, a 1xN ndarray
    Output:
        return a ap scalar and a cmc ndarray 1*N 
    """
    assert query_pid != -1
    cmc = np.zeros(gallery_pids.shape)
    ngood = np.sum((query_pid == gallery_pids) & (query_cam != gallery_cams))
    old_recall = 0.0
    old_precision = 1.0
    ap = 0.0
    intersect_size = 0.0
    j = 0
    good_now = 0
    njunk = 0
    for n, (p, c) in enumerate(zip(gallery_pids[0, :], gallery_cams[0, :])):
        flag = 0
        if p == query_pid and c != query_cam:
            flag = 1
            good_now = good_now + 1
            if good_now == 1:
                cmc[0,n-njunk:] = 1
        
        if seperate_cam:
            junk1 = c == query_cam
        else:
            junk1 = p == query_pid and c == query_cam
        junk2 = p == -1
        
        if junk1 or junk2:
            njunk = njunk + 1
            continue

        if flag == 1:
            intersect_size += 1.0

        recall = intersect_size/ngood
        precision = intersect_size/(j+1)
        ap = ap + (recall - old_recall)*((old_precision+precision)/2.)
        old_recall = recall
        old_precision = precision
        j = j + 1
        if good_now == ngood:
            break
    return ap, cmc

def evaluate_image_ss(dist_mat, pid, cam, **kwargs):
    """
    Input:
        dist_mat, pid, cam
    Output:
        result: dict
        result['ss']['CMC']
        result['ss']['mAP']
    """
    if kwargs.has_key('repeat_times'):
        T = kwargs['repeat_times']
    else:
        T = 1
    data = dict()
    for i, (p, c) in enumerate(zip(pid[0, :], cam[0, :])):
        if not data.has_key(p):
            data[p] = dict()
        if not data[p].has_key(c):
            data[p][c] = []
        data[p][c].append(i)
    # random T times for testing
    print('compute score for single shot.')
    for t in range(T):
        # sample query idx
        index = []
        for p in data.keys():
            for c in data[p].keys():
                index.append( np.random.choice(data[p][c]) )
        if t == 0:
            aps = np.zeros((1, T))
            cmcs = np.zeros((T, len(index)))
        dist_tmp = dist_mat[index, :][:, index]
        query_pid = pid[:, index]
        query_cam = cam[:, index]
        aps[0, t], cmcs[t, :] = compute_score(dist_tmp, query_pid, query_cam, query_pid, query_cam, True)
    mAP = np.mean(aps[:])
    CMC =  np.mean(cmcs, axis=0, keepdims=True)
    return mAP, CMC

def reid_evaluate_image_pids(feat_func, dataset, **kwargs):
    """ for each pid at each cam, set the first one as query and the rest as gallery.
        only using single-query for validation
        for each pid at each cam, selected the first one sample for validation
    """
    dataset.create_image_list_by_pids()
    feat, pid, cam, seq, frame, record = extract_feat(feat_func, dataset)
    return reid_evaluate_image_sequence_pids(feat, pid, cam, **kwargs)

def reid_evaluate_image_sequence_pids(feat, pid, cam, **kwargs):
    """
    shared by image-based and sequence-based person re-identification
    """
    # re-organize the data 
    result = dict()
    data = dict()
    for i, (p, c) in enumerate(zip(pid, cam)):
        if not data.has_key(p):
            data[p] = dict()
        if not data[p].has_key(c):
            data[p][c] = []
        data[p][c].append(i)
    # default: single query
    sq_flag = kwargs.has_key('eval_type') and 'sq' in kwargs['eval_type']
    sq_flag = sq_flag or not kwargs.has_key('eval_type')
    if sq_flag:
        # for each person at each camera, sample one image
        query_idx = []
        for p in data.keys():
            for c in data[p].keys():
                query_idx.append(data[p][c][0])
        gallery_pid = np.array(pid).reshape((1, len(pid)))
        gallery_cam = np.array(cam).reshape((1, len(cam)))
        gallery_feat = feat
        query_pid = gallery_pid[:, query_idx]
        query_cam = gallery_cam[:, query_idx]
        query_feat = feat[query_idx, :]
        print('compute distance for single query.')
        if kwargs.has_key('dist_type'):
            dist_mat = compute_dist(query_feat, gallery_feat, dist_type=kwargs['dist_type'], verbose=True)
        else:
            dist_mat = compute_dist(query_feat, gallery_feat, dist_type='euclidean_normL2', verbose=True)
        mAP, CMC = compute_score(dist_mat, query_pid, query_cam, gallery_pid, gallery_cam)
        result['sq'] = dict()
        result['sq']['mAP'] = mAP
        result['sq']['CMC'] = CMC
    # mutiple query 
    if kwargs.has_key('eval_type') and 'mq' in kwargs['eval_type']:
        # re-organize the query feature
        query_pid = []
        query_cam = []
        query_idx = []
        for p in data.keys():
            for c in data[p].keys():
                query_pid.append(p)
                query_cam.append(c)
                query_idx.append(data[p][c])
        # pooling the feat
        Q = len(query_idx)
        D = feat.shape[1]
        query_feat = np.zeros((Q, D))
        for i in range(len(query_idx)):
            query_feat[i,:] = feature_pooling(feat[query_idx[i], :], **kwargs)
        query_pid = np.array(query_pid).reshape((1, Q))
        query_cam = np.array(query_cam).reshape((1, Q))
        gallery_pid = np.array(pid).reshape((1, len(pid)))
        gallery_cam = np.array(cam).reshape((1, len(cam)))
        gallery_feat = feat
        print('compute distance for mutiple query.')
        if kwargs.has_key('dist_type'):
            dist_mat = compute_dist(query_feat, gallery_feat, dist_type=kwargs['dist_type'], verbose=True)
        else:
            dist_mat = compute_dist(query_feat, gallery_feat, dist_type='euclidean_normL2', verbose=True)
        print('compute score for mutiple query.')
        mAP, CMC = compute_score(dist_mat, query_pid, query_cam, gallery_pid, gallery_cam)
        result['mq'] = dict()
        result['mq']['mAP'] = mAP
        result['mq']['CMC'] = CMC
            
    # single shot, specially for cuhk03 val/test in old style
    if kwargs.has_key('eval_type') and 'ss' in kwargs['eval_type']:
        print('compute distance for single shot.')
        if kwargs.has_key('dist_type'):
            dist_mat = compute_dist(feat, feat, dist_type=kwargs['dist_type'], verbose=True)
        else:
            dist_mat = compute_dist(feat, feat, dist_type='euclidean_normL2', verbose=True)
        query_pid = np.array(pid).reshape((1, len(pid)))
        query_cam = np.array(cam).reshape((1, len(cam)))
        mAP, CMC = evaluate_image_ss(dist_mat, query_pid, query_cam, **kwargs)
        result['ss'] = dict()
        result['ss']['mAP'] = mAP
        result['ss']['CMC'] = CMC
    
    # mutiple shot specially for cuhk03 val/test in old style
    if kwargs.has_key('eval_type') and 'ms' in kwargs['eval_type']:
        query_pid = []
        query_cam = []
        query_idx = []
        for p in data.keys():
            for c in data[p].keys():
                query_pid.append(p)
                query_cam.append(c)
                query_idx.append(data[p][c])
        # pooling the feat
        Q = len(query_idx)
        D = feat.shape[1]
        query_feat = np.zeros((Q, D))
        for i in range(len(query_idx)):
            query_feat[i,:] = feature_pooling(feat[query_idx[i], :], **kwargs)
        query_pid = np.array(query_pid).reshape((1, Q))
        query_cam = np.array(query_cam).reshape((1, Q))
        print('compute distance for mutiple shot.')
        if kwargs.has_key('dist_type'):
            dist_mat = compute_dist(query_feat, query_feat, dist_type=kwargs['dist_type'], verbose=True)
        else:
            dist_mat = compute_dist(query_feat, query_feat, dist_type='euclidean_normL2', verbose=True)
        print('compute score for mutiple shot.')
        mAP, CMC = evaluate_image_ss(dist_mat, query_pid, query_cam, repated_times=1)
        result['ms'] = dict()
        result['ms']['mAP'] = mAP
        result['ms']['CMC'] = CMC
    return result

def reid_evaluate_image_viper(feat_func, dataset, **kwargs):
    """ for viper, always the pids-based evaluatation. """
    return reid_evaluate_image_pids(feat_func, dataset, **kwargs)

def reid_evaluate_image_cuhk03_old(feat_func, dataset, **kwargs):
    return reid_evaluate_image_pids(feat_func, dataset, **kwargs)

def reid_evaluate_image_cuhk03(feat_func, dataset, **kwargs):
    if kwargs.has_key('cuhk03_new') and kwargs['cuhk03_new']:
        return reid_evaluate_image_cuhk03_new(feat_func, dataset, **kwargs)
    else:
        return reid_evaluate_image_cuhk03_old(feat_func, dataset, **kwargs)

def reid_evaluate_image_fixed_query_gallery(feat_func, dataset, **kwargs):
    """ fixed query, gallery
        if using mutiple query, using fixed groundtruth or gallery
    """
    
    print('Extracting features for fixed query.')
    dataset.create_image_list_by_fixed_query()
    query_feat, query_pid, query_cam, query_seq, query_frame, query_record = \
        extract_feat(feat_func, dataset)

    print('Extracting features for fixed gallery.')
    dataset.create_image_list_by_fixed_gallery()
    gallery_feat, gallery_pid, gallery_cam, gallery_seq, gallery_frame, gallery_record = \
        extract_feat(feat_func, dataset)
    
    # mutiple query
    if kwargs.has_key('eval_type') and 'mq' in kwargs['eval_type']:
        if dataset.dataset.has_key('image_gt'):
            print('Extracting features for fixed groundtruth in mutiple query.')
            dataset.create_image_list_by_fixed_groundtruth()
            gt_feat, gt_pid, gt_cam, gt_seq, gt_frame, gt_record = \
                extract_feat(feat_func, dataset)
            # GT = len(gt_pid)
            # gt_pid = np.array(gt_pid).reshape((1, GT))
            # gt_cam = np.array(gt_cam).reshape((1, GT))
        else:
            gt_feat = gallery_feat
            gt_pid = gallery_pid
            gt_cam = gallery_cam
    else:
        gt_pid = []
        gt_pid = []
        gt_cam = []
        gt_feat = None

    return  reid_evaluate_image_sequence_fixed_query_gallery_groundtruth(query_feat, query_pid, query_cam, \
        gallery_feat, gallery_pid, gallery_cam, gt_feat, gt_pid, gt_cam, **kwargs)

def reid_evaluate_image_sequence_fixed_query_gallery_groundtruth(query_feat, query_pid, query_cam, \
    gallery_feat, gallery_pid, gallery_cam, gt_feat, gt_pid, gt_cam, **kwargs):
    """
    Output:
        result
    """
    result = dict()
    # single query
    Q = len(query_pid)
    G = len(gallery_pid)
    D = query_feat.shape[1]
    query_pid = np.array(query_pid).reshape((1, Q))
    query_cam = np.array(query_cam).reshape((1, Q))
    gallery_pid = np.array(gallery_pid).reshape((1, G))
    gallery_cam = np.array(gallery_cam).reshape((1, G))

    print('compute distance for single query.')
    if kwargs.has_key('dist_type'):
        dist_mat = compute_dist(query_feat, gallery_feat, dist_type=kwargs['dist_type'], verbose=True)
    else:
        dist_mat = compute_dist(query_feat, gallery_feat, dist_type='euclidean_normL2', verbose=True)
    
    print('compute score for single query.')
    mAP, CMC = compute_score(dist_mat, query_pid, query_cam, gallery_pid, gallery_cam)
    result['sq'] = dict()
    result['sq']['mAP'] = mAP
    result['sq']['CMC'] = CMC

    if kwargs.has_key('rerank_k1'):
        k1 = kwargs['rerank_k1']
    else:
        k1 = 20

    if kwargs.has_key('rerank_k2'):
        k2 = kwargs['rerank_k2']
    else:
        k2 = 6
       
    if kwargs.has_key('rerank_lambda'):
        lambda_value = kwargs['rerank_lambda']
    else:
        lambda_value = 0.3

    if kwargs.has_key('rerank') and kwargs['rerank']:
        q_g_dist = dist_mat
        print('compute distance for single query rerank.')
        if kwargs.has_key('dist_type'):
            q_q_dist = compute_dist(query_feat, query_feat, dist_type=kwargs['dist_type'], verbose=True)
            g_g_dist = compute_dist(gallery_feat, gallery_feat, dist_type=kwargs['dist_type'], verbose=True)
        else:
            q_q_dist = compute_dist(query_feat, query_feat, dist_type='euclidean_normL2', verbose=True)
            g_g_dist = compute_dist(gallery_feat, gallery_feat, dist_type='euclidean_normL2', verbose=True)
        rerank_sq_dist = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1, k2, lambda_value)
        print('compute score for single query rerank.')
        mAP, CMC = compute_score(rerank_sq_dist, query_pid, query_cam, gallery_pid, gallery_cam)
        result['sq_rerank'] = dict()
        result['sq_rerank']['mAP'] = mAP
        result['sq_rerank']['CMC'] = CMC
     
    # only single query
    GT = len(gt_pid)
    if GT == 0:
        return result
    
    mquery_feat = np.zeros((Q, D))
    gt_pid = np.array(gt_pid).reshape((1, GT))
    for i, (p, c) in enumerate(zip(query_pid[0, :], query_cam[0, :])):
        # idx is a 1*G bool array
        idx = ((p == gt_pid).astype(float) + (c == gt_cam).astype(float)) == 2
        mquery_feat[i, :] = feature_pooling(gt_feat[idx[0, :], :], **kwargs)
    
    print('compute distance for mutiple query.')
    if kwargs.has_key('dist_type'):
        dist_mat = compute_dist(mquery_feat, gallery_feat, dist_type=kwargs['dist_type'], verbose=True)
    else:
        dist_mat = compute_dist(mquery_feat, gallery_feat, dist_type='euclidean_normL2', verbose=True)
    print('compute score for mutiple query.')
    mAP, CMC = compute_score(dist_mat, query_pid, query_cam, gallery_pid, gallery_cam)
    result['mq'] = dict()
    result['mq']['mAP'] = mAP
    result['mq']['CMC'] = CMC

    # rerank sq
    if kwargs.has_key('rerank') and kwargs['rerank']:
        mq_g_dist = dist_mat
        print('compute distance for mutiple query rerank.')
        if kwargs.has_key('dist_type'):
            mq_mq_dist = compute_dist(mquery_feat, mquery_feat, dist_type=kwargs['dist_type'], verbose=True)
        else:
            mq_mq_dist = compute_dist(mquery_feat, mquery_feat, dist_type='euclidean_normL2', verbose=True)
        rerank_mq_dist = re_ranking(mq_g_dist, mq_mq_dist, g_g_dist, k1, k2, lambda_value)
        print('compute score for mutiple query rerank.')
        mAP, CMC = compute_score(rerank_mq_dist, query_pid, query_cam, gallery_pid, gallery_cam)
        result['mq_rerank'] = dict()
        result['mq_rerank']['mAP'] = mAP
        result['mq_rerank']['CMC'] = CMC
    
    return result

def reid_evaluate_image_cuhk03_new(feat_func, dataset, **kwargs):
    return reid_evaluate_image_fixed_query_gallery(feat_func, dataset, **kwargs)

def reid_evaluate_image_market1501(feat_func, dataset, **kwargs):
    return reid_evaluate_image_fixed_query_gallery(feat_func, dataset, **kwargs)

def reid_evaluate_image_duke(feat_func, dataset, **kwargs):
    return reid_evaluate_image_fixed_query_gallery(feat_func, dataset, **kwargs)

def reid_evaluate_image_rap2(feat_func, dataset, **kwargs):
    return reid_evaluate_image_fixed_query_gallery(feat_func, dataset, **kwargs)

def reid_evaluate_sequence_pids(feat_func, dataset, **kwargs):
    """ pool the feature using sequence
    Input: 
        feat_func, a call able model
        dataset, an object of re-identification Dataset class
    """
    result = dict()
    dataset.create_image_list_by_pids()
    feat, pid, cam, seq, frame, record = extract_feat(feat_func, dataset)
    # re-organize the data using id, cam, seq
    data = dict()
    N_seq = 0 
    for i, (p, c, s) in enumerate(zip(pid, cam, seq)):
        if not data.has_key(p):
            data[p] = dict()
        if not data[p].has_key(c):
            data[p][c] = dict()
        if not data[p][c].has_key(s):
            data[p][c][s] = []
            N_seq = N_seq + 1 
        data[p][c][s].append(i)
    # sort the sequence, re-compute the sequence-based feature
    spid = []
    scam = []
    sseq = []
    cnt = 0
    for p in data.keys():
        for c in data[p].keys():
            for s in data[p][c]:
                tmp = np.sort(data[p][c][s])
                if cnt == 0:
                    sfeat_tmp = feature_pooling(feat[tmp, :], **kwargs) 
                    spid.append(p)
                    scam.append(c)
                    scam.append(s)
                    L = sfeat_tmp.shape[1]
                    sfeat = np.zeros((N_seq, L))
                    sfeat[cnt, :] = sfeat_tmp
                    cnt = cnt + 1
                spid.append(p)
                scam.append(c)
                scam.append(s)
                sfeat[cnt, :] = feature_pooling(feat[tmp, :], **kwargs)
    # re-compute the sequence-based feature
    return reid_evaluate_image_sequence_pids(sfeat, spid, scam, **kwargs)
    
def reid_evaluate_sequence_mars(feat_func, dataset, **kwargs):
    # adapted to the reid_evaluate_image_sequence_fixed_query_gallery_groundtruth

    # for mars dataset, first create image list using test identites.
    # then using fixed query/gallery tracklets for evaluatation
    dataset.create_image_list_by_pids()
    feat, pid, cam, seq, frame, record = extract_feat(feat_func, dataset)
    # re-organize the data using id, cam, seq
    data = dict()
    N_seq = 0 
    for i, (p, c, s) in enumerate(zip(pid, cam, seq)):
        if not data.has_key(p):
            data[p] = dict()
        if not data[p].has_key(c):
            data[p][c] = dict()
        if not data[p][c].has_key(s):
            data[p][c][s] = []
            N_seq = N_seq + 1 
        data[p][c][s].append(i)
    # sort the sequence, re-compute the sequence-based feature
    # spid = []
    # scam = []
    # sseq = []
    # extract the fixed query/gallery tracklets feature 
    track_pid_q = dataset.dataset['track_pid_q']
    track_cam_q = dataset.dataset['track_cam_q']
    track_seq_q = dataset.dataset['track_seq_q']
    track_pid_g = dataset.dataset['track_pid_g']
    track_cam_g = dataset.dataset['track_cam_g']
    track_seq_g = dataset.dataset['track_seq_g']
    N_q = len(track_pid_q)
    N_g = len(track_pid_g)
    track_pid_q = np.array(track_pid_q).reshape((1, N_q))
    track_cam_q = np.array(track_cam_q).reshape((1, N_q))
    track_seq_q = np.array(track_seq_q).reshape((1, N_q))
    track_pid_g = np.array(track_pid_g).reshape((1, N_g))
    track_cam_g = np.array(track_cam_g).reshape((1, N_g))
    track_seq_g = np.array(track_seq_g).reshape((1, N_g))

    query_idx = []
    gallery_idx = []
    query_pid = []
    query_cam = []
    gallery_pid = []
    gallery_cam = []
    cnt = 0
    for p in data.keys():
        for c in data[p].keys():
            for s in data[p][c]:
                tmp = np.sort(data[p][c][s])
                sfeat_tmp = feature_pooling(feat[tmp, :], **kwargs) 
                if cnt == 0:
                    L = sfeat_tmp.shape[1]
                    sfeat = np.zeros((N_seq, L))
                # spid.append(p)
                # scam.append(c)
                # scam.append(s)
                sfeat[cnt, :] = feature_pooling(feat[tmp, :], **kwargs)
                # process the query/gallery index
                flag = np.sum((track_pid_q == p) & (track_cam_q == c) & (track_seq_q == s))
                if flag:
                    query_idx.append(cnt)
                    query_pid.append(p)
                    query_cam.append(c)
                flag = np.sum((track_pid_g == p) & (track_cam_g == c) & (track_seq_g == s))
                if flag:
                    gallery_idx.append(cnt)
                    gallery_pid.append(p)
                    gallery_cam.append(c)
                cnt = cnt + 1
    assert N_seq == cnt
    query_feat = sfeat[query_idx, :]
    gallery_feat = sfeat[gallery_idx, :]
    # for groundtruth
    gt_feat = gallery_feat
    gt_pid = gallery_pid
    gt_cam = gallery_cam
    return reid_evaluate_image_sequence_fixed_query_gallery_groundtruth( \
            query_feat, query_pid, query_cam, gallery_feat, gallery_pid, gallery_cam, \
            gt_feat, gt_pid, gt_cam, **kwargs)

def reid_evaluate_image(feat_func, dataset, **kwargs):
    """
    image-based person re-identification
    """
    if dataset.split == 'val':
        return reid_evaluate_image_pids(feat_func, dataset, **kwargs)

    if dataset.dataset['description'] == 'viper':
        return reid_evaluate_image_viper(feat_func, dataset, **kwargs)

    if dataset.dataset['description'] == 'cuhk03':
        return reid_evaluate_image_cuhk03(feat_func, dataset, **kwargs)
    
    if dataset.dataset['description'] == 'market1501':
        return reid_evaluate_image_market1501(feat_func, dataset, **kwargs)
    
    if dataset.dataset['description'] == 'duke':
        return reid_evaluate_image_duke(feat_func, dataset, **kwargs)

    if dataset.dataset['description'] == "rap2":
        return reid_evaluate_image_rap2(feat_func, dataset, **kwargs)

    print('The dataset: {} does not support evaluatation.'%(dataset.dataset['description']))
    raise ValueError
    
def reid_evaluate_sequence(feat_func, dataset, **kwargs):
    
    if dataset.split == 'val':
        return reid_evaluate_sequence_pids(feat_func, dataset, **kwargs)

    if dataset.dataset['description'] == 'mars':
        return reid_evaluate_sequence_mars(feat_func, dataset, **kwargs)
    
    print('The dataset: {} does not support evaluatation.'%(dataset.dataset['description']))
    raise ValueError

# copy from houjing huang
def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist
