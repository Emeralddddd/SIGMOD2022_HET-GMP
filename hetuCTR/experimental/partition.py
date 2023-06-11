import numpy as np
import scipy.sparse as sp
import time
import argparse
import os.path as osp

import hetuCTR_partition
def load_criteo_data():
    fname = "/data/1/zhen/dac/sparse_feats.npy"
    assert osp.exists(fname)
    data = np.load(fname)
    if not data.data.c_contiguous:
        data = np.ascontiguousarray(data)
    return data

def get_comm_mat(nrank, ngpus):
    mat = np.ones([nrank, nrank])
    for i in range(nrank):
        for j in range(nrank):
            if i == j:
                mat[i, j] = 0
            elif i // ngpus == j // ngpus:
                mat[i, j] = 0.1
            else:
                mat[i, j] = 1
    return mat

def direct_partition(data, nparts, ngpus, batch_size, rerun, output):
    start = time.time()
    mat = get_comm_mat(nparts, ngpus)
    partition = hetuCTR_partition.partition(data, mat, nparts, batch_size, 0.01)

    cost = partition.get_communication()
    print("Initial cost : {}".format(np.multiply(cost, mat).sum()))
    for i in range(rerun):
        t_start = time.time()
        partition.refine_data()
        t_mid = time.time()
        partition.refine_embed()
        print("Refine Time : ",t_mid-t_start, " ",time.time()-t_mid)
        cost = partition.get_communication()
        print("Refine round {} : {}".format(i, np.multiply(cost, mat).sum()))
        print("Data :", partition.get_data_cnt())
        print("Embed:", partition.get_embed_cnt())
        print("In   :", np.sum(cost, axis=0))
        print("Out  :", np.sum(cost, axis=1))
        print(cost.astype(np.int))
    item_partition, idx_partition = partition.get_result()
    print("Partition Time : ", time.time()-start)
    start = time.time()

    arr_dict = {"embed_partition" : idx_partition, "data_partition" : item_partition}
    priority = partition.get_priority()
    for i in range(nparts):
        idxs = np.where(idx_partition==i)[0]
        priority[i][idxs] = -1 # remove embedding that has been stored
        arr = np.argsort(priority[i])[len(idxs):][ : : -1]
        arr_dict[str(i)] = arr
    arr_dict["priority"] = priority
    print("Sort priority Time : ", time.time()-start)

    if output != "":
        np.savez(output, **arr_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrank", "-n" , type=int, default=4)
    parser.add_argument("--batch_size", "-b" , type=int, default=1)
    parser.add_argument("--rerun", "-r" , type=int, default=5)
    parser.add_argument("--ngpus", "-g" , type=int, default=1)
    parser.add_argument("--output", "-o" , type=str, default="/data/1/zhen/dac/partition/temp.npz")
    args = parser.parse_args()
    start = time.time()
    data = load_criteo_data()
    print("Load Data Time : ", time.time()-start)
    # np.random.shuffle(data)
    direct_partition(data[:1000000], args.nrank, args.ngpus, args.batch_size, args.rerun,"/data/1/zhen/dac/partition/temp.npz")
    # for i in range(1,80):
    #     print(i)
    #     direct_partition(data[(i-1)*1000000:i * 1000000], args.nrank, args.ngpus, args.batch_size, args.rerun,"/data/1/zhen/criteo-tb/partition/new/day0_{}m.npz".format(i))