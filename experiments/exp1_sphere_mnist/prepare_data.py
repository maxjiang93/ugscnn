'''Module to generate the spherical mnist data set'''

import sys; sys.path.append("../../meshcnn")
import gzip
import pickle
import numpy as np
import argparse
from torchvision import datasets
from utils import interp_r2tos2
from tqdm import tqdm


NORTHPOLE_EPSILON = 1e-3

def meshgrid(b, grid_type='Driscoll-Healy'):
    return np.meshgrid(*linspace(b, grid_type), indexing='ij')


def linspace(b, grid_type='Driscoll-Healy'):
    if grid_type == 'Driscoll-Healy':
        beta = np.arange(2 * b) * np.pi / (2. * b)
        alpha = np.arange(2 * b) * np.pi / b
    else:
        raise ValueError('Unknown grid_type:' + grid_type)
    return beta, alpha


def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    # http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    """

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M


def rotate_grid(rot, grid):
    x, y, z = grid
    xyz = np.array((x, y, z))
    x_r, y_r, z_r = np.einsum('ij,jab->iab', rot, xyz)
    return x_r, y_r, z_r


def get_projection_grid(b, grid_type="Driscoll-Healy"):
    ''' returns the spherical grid in euclidean
    coordinates, where the sphere's center is moved
    to (0, 0, 1)'''
    theta, phi = meshgrid(b=b, grid_type=grid_type)
    x_ = np.sin(theta) * np.cos(phi)
    y_ = np.sin(theta) * np.sin(phi)
    z_ = np.cos(theta)
    return x_, y_, z_


def project_sphere_on_xy_plane(grid, projection_origin):
    ''' returns xy coordinates on the plane
    obtained from projecting each point of
    the spherical grid along the ray from
    the projection origin through the sphere '''

    sx, sy, sz = projection_origin
    x, y, z = grid
    z = z.copy() + 1

    t = -z / (z - sz)
    qx = t * (x - sx) + x
    qy = t * (y - sy) + y

    xmin = 1/2 * (-1 - sx) + -1
    ymin = 1/2 * (-1 - sy) + -1

    # ensure that plane projection
    # ends up on southern hemisphere
    rx = (qx - xmin) / (2 * np.abs(xmin))
    ry = (qy - ymin) / (2 * np.abs(ymin))

    return rx, ry


def sample_within_bounds(signal, x, y, bounds):
    ''' '''
    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)

    if len(signal.shape) > 2:
        sample = np.zeros((signal.shape[0], x.shape[0], x.shape[1]))
        sample[:, idxs] = signal[:, x[idxs], y[idxs]]
    else:
        sample = np.zeros((x.shape[0], x.shape[1]))
        sample[idxs] = signal[x[idxs], y[idxs]]
    return sample


def sample_bilinear(signal, rx, ry):
    ''' '''

    signal_dim_x = signal.shape[1]
    signal_dim_y = signal.shape[2]

    rx *= signal_dim_x
    ry *= signal_dim_y

    # discretize sample position
    ix = rx.astype(int)
    iy = ry.astype(int)

    # obtain four sample coordinates
    ix0 = ix - 1
    iy0 = iy - 1
    ix1 = ix + 1
    iy1 = iy + 1

    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    # linear interpolation in x-direction
    fx1 = (ix1-rx) * signal_00 + (rx-ix0) * signal_10
    fx2 = (ix1-rx) * signal_01 + (rx-ix0) * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry) * fx1 + (ry - iy0) * fx2


def project_2d_on_sphere(signal, grid, projection_origin=None):
    ''' '''
    if projection_origin is None:
        projection_origin = (0, 0, 2 + NORTHPOLE_EPSILON)

    rx, ry = project_sphere_on_xy_plane(grid, projection_origin)
    sample = sample_bilinear(signal, rx, ry)

    # ensure that only south hemisphere gets projected
    sample *= (grid[2] <= 1).astype(np.float64)

    # rescale signal to [0,1]
    sample_min = sample.min(axis=(1, 2)).reshape(-1, 1, 1)
    sample_max = sample.max(axis=(1, 2)).reshape(-1, 1, 1)

    sample = (sample - sample_min) / (sample_max - sample_min)
    sample *= 255
    sample = sample.astype(np.uint8)

    return sample


def main():
    ''' '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--bandwidth",
                        help="the bandwidth of the S2 signal",
                        type=int,
                        default=30,
                        required=False)
    parser.add_argument("--noise",
                        help="the rotational noise applied on the sphere",
                        type=float,
                        default=1.0,
                        required=False)
    parser.add_argument("--chunk_size",
                        help="size of image chunk with same rotation",
                        type=int,
                        default=500,
                        required=False)
    parser.add_argument("--mnist_data_folder",
                        help="folder for saving the mnist data",
                        type=str,
                        default="MNIST_data",
                        required=False)
    parser.add_argument("--output_file",
                        help="file for saving the data output (.gz file)",
                        type=str,
                        default="mnist_ico3.gzip",
                        required=False)
    parser.add_argument("--no_rotate_train",
                        help="do not rotate train set",
                        dest='no_rotate_train', action='store_true')
    parser.add_argument("--no_rotate_test",
                        help="do not rotate test set",
                        dest='no_rotate_test', action='store_true')
    parser.add_argument("--mesh_file",
                        help="path to mesh file",
                        type=str,
                        default="mesh_files/icosphere_3.pkl")
    parser.add_argument("--direction", 
                        help="projection direction [NP/EQ] : North Pole / Equator",
                        type=str,
                        choices=["NP", "EQ"],
                        default="EQ")
    
    args = parser.parse_args()

    print("getting mnist data")
    trainset = datasets.MNIST(root=args.mnist_data_folder, train=True, download=True)
    testset = datasets.MNIST(root=args.mnist_data_folder, train=False, download=True)
    mnist_train = {}
    mnist_train['images'] = trainset.train_data.numpy()
    mnist_train['labels'] = trainset.train_labels.numpy()
    mnist_test = {}
    mnist_test['images'] = testset.test_data.numpy()
    mnist_test['labels'] = testset.test_labels.numpy()
    
    grid = get_projection_grid(b=args.bandwidth)

    # result
    dataset = {}

    no_rotate = {"train": args.no_rotate_train, "test": args.no_rotate_test}

    for label, data in zip(["train", "test"], [mnist_train, mnist_test]):

        print("projecting {0} data set".format(label))
        current = 0
        signals = data['images'].reshape(-1, 28, 28).astype(np.float64)
        n_signals = signals.shape[0]
        projections = np.ndarray(
            (signals.shape[0], 2 * args.bandwidth, 2 * args.bandwidth),
            dtype=np.uint8)

        while current < n_signals:

            if not no_rotate[label]:
                rot = rand_rotation_matrix(deflection=args.noise)
                rotated_grid = rotate_grid(rot, grid)
            else:
                rotated_grid = grid

            idxs = np.arange(current, min(n_signals,
                                          current + args.chunk_size))
            chunk = signals[idxs]
            projections[idxs] = project_2d_on_sphere(chunk, rotated_grid)
            current += args.chunk_size
            print("\r{0}/{1}".format(current, n_signals), end="")
        print("")
        dataset[label] = {
            'images': projections,
            'labels': data['labels']
        }

    x_train = dataset['train']['images']
    x_test = dataset['test']['images']
    y_train = dataset['train']['labels']
    y_test = dataset['test']['labels']
    p = pickle.load(open(args.mesh_file, "rb"))
    V = p['V']
    F = p['F']
    
    # whether to project to NP (north pole) or EQ (equator)
    cos45 = np.cos(np.pi/2)
    sin45 = np.sin(np.pi/2)
    if args.direction == "EQ":
        x_rot_mat = np.array([[1, 0, 0],[0, cos45, -sin45],[0,sin45, cos45]])
        V = V.dot(x_rot_mat)

    x_train_s2 = []
    print("Converting training set...")
    for i in tqdm(range(x_train.shape[0])):
        x_train_s2.append(interp_r2tos2(x_train[i], V))

    x_test_s2 = []
    print("Converting test set...")
    for i in tqdm(range(x_test.shape[0])):
        x_test_s2.append(interp_r2tos2(x_test[i], V))

    x_train_s2 = np.stack(x_train_s2, axis=0)
    x_test_s2 = np.stack(x_test_s2, axis=0)
    
    d = {"train_inputs": x_train_s2,
     "train_labels": y_train,
     "test_inputs": x_test_s2,
     "test_labels": y_test}
    
    pickle.dump(d, gzip.open(args.output_file, "wb"))


if __name__ == '__main__':
    main()
