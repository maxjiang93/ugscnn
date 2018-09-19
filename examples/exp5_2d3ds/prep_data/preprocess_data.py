import numpy as np
import sys
sys.path.append("../../../meshcnn")
from utils import interp_r2tos2
import argparse
import os
from glob import glob
from joblib import Parallel, delayed
import pickle
from scipy import misc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num",
                        help="number of samples to process",
                        default=-1,
                        type=int)
    parser.add_argument("-r", "--random",
                        help="random draw of samples",
                        action='store_true')
    parser.add_argument("-s", "--random_seed",
                        help="random seed",
                        type=int,
                        default=0)
    parser.add_argument("-d", "--data_directory",
                        help="path to data directory",
                        type=str,
                        default="/global/cscratch1/sd/maxjiang/im2pano3d/data/mpv3")
    parser.add_argument("-o", "--output_directory",
                        help="output directory",
                        type=str,
                        default="/global/cscratch1/sd/maxjiang/matterport3d_sphere")
    parser.add_argument("-m", "--mesh_file",
                        help="path to template mesh file",
                        type=str,
                        default="../../../mesh_files/icosphere_7.pkl")
    parser.add_argument("-j", "--jobs",
                        help="number of parallel jobs to run",
                        type=int,
                        default=64)
    args = parser.parse_args()

    # make output directory
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    flist = glob(os.path.join(args.data_directory, "*/pano/*"))
    flist = [p for p in flist if ("_d_" in p) or ("_i_" in p) or ("_label13" in p)]

    if not len(flist) > 0:
        print("[!] Wrong data directory.")
        assert(0)
    if len(flist) < args.num:
        print("[!] Number of samples less than avaliable data files")
        assert(0)
    if args.num < 1:
        args.num = len(flist)

    # random draw files
    if args.random:
        np.random.seed(args.random_seed)
        seq = np.random.permutation(len(flist)).astype(np.int)
    else:
       seq = np.arange(len(flist)).astype(np.int)
    fl = np.array(flist)[seq[:args.num]]

    if args.num == len(fl):
        fin_list = glob(os.path.join(args.output_directory, "*/pano/*.npz"))
        # l1 = [os.path.basename(p).replace("-mesh.npz", "") for p in fin_list]
        l1 = [os.path.basename(p).replace(".npz", "") for p in fin_list]
        l2 = [os.path.basename(p).split('_')[0] for p in fl]
        l1 = np.array(l1)
        l2 = np.array(l2)
        unfinish_mask = ~np.in1d(l2, l1)
        fl = fl[unfinish_mask]

    fl = np.unique(np.array([p.split("_")[0] for p in fl]))
    rgb_tail = "_i_r0.4.jpg"
    depth_tail = "_d_r0.4.png"
    sem_tail = "_label13.png"

    # load mesh file
    pkl = pickle.load(open(args.mesh_file, "rb"))
    V = pkl['V']

    # process data
    def worker_exec(file):
        """
        execute data processing per worker
        """
        rgb = misc.imread(file+rgb_tail) / 255.0
        depth = misc.imread(file+depth_tail) * 0.25 / 1000.0
        sem = misc.imread(file+sem_tail); sem[sem == 255] = 14

        rgbd = np.concatenate((rgb, np.expand_dims(depth, -1)), -1)

        rgbd_intp = interp_r2tos2(rgbd, V, "linear", np.float32) # linearly interpolate RGBD values
        sem_intp = interp_r2tos2(sem, V, "nearest", np.uint8) # nearest neighbor intepolation for label

        # save
        fname = '/'.join(file.split('/')[-3:])+".npz"
        fname = os.path.join(args.output_directory, fname)
        if not os.path.exists(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))

        rgbd_intp = np.moveaxis(rgbd_intp, -1, 0) # change into shape of (n_channel, height, width)
        np.savez(fname, data=rgbd_intp, labels=sem_intp)

    from pdb import set_trace; set_trace()
    worker_exec(fl[0])
    # Parallel(n_jobs=args.jobs)(delayed(worker_exec)(fname) for fname in fl)

if __name__ == '__main__':
    main()


