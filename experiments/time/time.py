import argparse
from model import Model, Model_tiny
import torch
import numpy as np
from time import time

mesh_sizes = [12, 42, 162, 642, 2562, 10242, 40962, 163842]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--feat", type=int, default=16)
    parser.add_argument("--ty", action='store_true')
    parser.add_argument('--neval', type=int, default=64, help='Number of evaluations to run for timing.')
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--sp_mesh_dir", type=str, default="../../mesh_files")
    parser.add_argument("--sp_mesh_level", type=int, default=5)
    args = parser.parse_args()
    print(args)

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    if not args.ty:
        model = Model(40, mesh_folder=args.sp_mesh_dir, feat=args.feat)
    else:
        model = Model_tiny(40, mesh_folder=args.sp_mesh_dir, feat=args.feat)
    model.to(device)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    print("{} paramerters in the last layer".format(sum(x.numel() for x in model.out_layer.parameters())))

    model.eval()
    total_time = []
    for i in range(args.neval):
        data = np.random.rand(args.batch_size, 6, mesh_sizes[args.sp_mesh_level]).astype(np.float32)
        data = torch.tensor(data, requires_grad=False).to(device)
        t0 = time()
        _ = model(data)
        total_time.append(time() - t0)
    print("Average Time per Batch: {} Secs".format(np.mean(total_time[10:])))
