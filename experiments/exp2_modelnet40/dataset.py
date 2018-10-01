# pylint: disable=E1101,R,C
import csv
import glob
import os
import re
import numpy as np
import torch
import torch.utils.data
import trimesh
import logging
import pickle

logging.getLogger('pyembree').disabled = True


def rotmat(a, b, c, hom_coord=False):  # apply to mesh using mesh.apply_transform(rotmat(a,b,c, True))
    """
    Create a rotation matrix with an optional fourth homogeneous coordinate

    :param a, b, c: ZYZ-Euler angles
    """
    def z(a):
        return np.array([[np.cos(a), np.sin(a), 0, 0],
                         [-np.sin(a), np.cos(a), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def y(a):
        return np.array([[np.cos(a), 0, np.sin(a), 0],
                         [0, 1, 0, 0],
                         [-np.sin(a), 0, np.cos(a), 0],
                         [0, 0, 0, 1]])

    r = z(a).dot(y(b)).dot(z(c))  # pylint: disable=E1101
    if hom_coord:
        return r
    else:
        return r[:3, :3]


def make_sgrid(b, alpha, beta, gamma):
    from lie_learn.spaces import S2

    theta, phi = S2.meshgrid(b=b, grid_type='SOFT')
    sgrid = S2.change_coordinates(np.c_[theta[..., None], phi[..., None]], p_from='S', p_to='C')
    sgrid = sgrid.reshape((-1, 3))

    R = rotmat(alpha, beta, gamma, hom_coord=False)
    sgrid = np.einsum('ij,nj->ni', R, sgrid)

    return sgrid


def render_model(mesh, sgrid):

    # Cast rays
    # triangle_indices = mesh.ray.intersects_first(ray_origins=sgrid, ray_directions=-sgrid)
    index_tri, index_ray, loc = mesh.ray.intersects_id(
        ray_origins=sgrid, ray_directions=-sgrid, multiple_hits=False, return_locations=True)
    loc = loc.reshape((-1, 3))  # fix bug if loc is empty

    # Each ray is in 1-to-1 correspondence with a grid point. Find the position of these points
    grid_hits = sgrid[index_ray]
    grid_hits_normalized = grid_hits / np.linalg.norm(grid_hits, axis=1, keepdims=True)

    # Compute the distance from the grid points to the intersection pionts
    dist = np.linalg.norm(grid_hits - loc, axis=-1)

    # For each intersection, look up the normal of the triangle that was hit
    normals = mesh.face_normals[index_tri]
    normalized_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Construct spherical images
    dist_im = np.ones(sgrid.shape[0])
    dist_im[index_ray] = dist
    # dist_im = dist_im.reshape(theta.shape)

    # shaded_im = np.zeros(sgrid.shape[0])
    # shaded_im[index_ray] = normals.dot(light_dir)
    # shaded_im = shaded_im.reshape(theta.shape) + 0.4

    n_dot_ray_im = np.zeros(sgrid.shape[0])
    # n_dot_ray_im[index_ray] = np.abs(np.einsum("ij,ij->i", normals, grid_hits_normalized))
    n_dot_ray_im[index_ray] = np.einsum("ij,ij->i", normalized_normals, grid_hits_normalized)

    nx, ny, nz = normalized_normals[:, 0], normalized_normals[:, 1], normalized_normals[:, 2]
    gx, gy, gz = grid_hits_normalized[:, 0], grid_hits_normalized[:, 1], grid_hits_normalized[:, 2]
    wedge_norm = np.sqrt((nx * gy - ny * gx) ** 2 + (nx * gz - nz * gx) ** 2 + (ny * gz - nz * gy) ** 2)
    n_wedge_ray_im = np.zeros(sgrid.shape[0])
    n_wedge_ray_im[index_ray] = wedge_norm

    # Combine channels to construct final image
    # im = dist_im.reshape((1,) + dist_im.shape)
    im = np.stack((dist_im, n_dot_ray_im, n_wedge_ray_im), axis=0)

    return im


def rnd_rot():
    a = np.random.rand() * 2 * np.pi
    z = np.random.rand() * 2 - 1
    c = np.random.rand() * 2 * np.pi
    rot = rotmat(a, np.arccos(z), c, True)
    return rot


class ToMesh:
    def __init__(self, random_rotations=False, random_translation=0):
        self.rot = random_rotations
        self.tr = random_translation

    def __call__(self, path):
        mesh = trimesh.load_mesh(path)
        mesh.remove_degenerate_faces()
        mesh.fix_normals()
        mesh.fill_holes()
        mesh.remove_duplicate_faces()
        mesh.remove_infinite_values()
        mesh.remove_unreferenced_vertices()

        mesh.apply_translation(-mesh.centroid)

        r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
        mesh.apply_scale(1 / r)

        if self.tr > 0:
            tr = np.random.rand() * self.tr
            rot = rnd_rot()
            mesh.apply_transform(rot)
            mesh.apply_translation([tr, 0, 0])

            if not self.rot:
                mesh.apply_transform(rot.T)

        if self.rot:
            mesh.apply_transform(rnd_rot())

        r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
        mesh.apply_scale(0.99 / r)

        return mesh

    def __repr__(self):
        return self.__class__.__name__ + '(rotation={0}, translation={1})'.format(self.rot, self.tr)


class ProjectOnSphere:
    def __init__(self, meshfile, dataset, normalize=True):
        self.meshfile = meshfile
        pkl = pickle.load(open(meshfile, "rb"))
        self.sgrid = pkl["V"]
        self.level = int(meshfile.split('_')[-1].split('.')[0])
        self.pts = self.sgrid.shape[0]
        self.normalize = normalize
        assert(dataset in ["modelnet10", "modelnet40"])
        self.dataset = dataset

    def __call__(self, mesh):
        im = render_model(mesh, self.sgrid)  # shape 3_channels x #v

        from scipy.spatial.qhull import QhullError  # pylint: disable=E0611
        try:
            convex_hull = mesh.convex_hull
        except QhullError:
            convex_hull = mesh

        hull_im = render_model(convex_hull, self.sgrid)

        im = np.concatenate([im, hull_im], axis=0)
        assert len(im) == 6

        # take absolute value of normal
        im[1] = np.absolute(im[1])
        im[4] = np.absolute(im[4])

        if self.normalize and self.dataset == 'modelnet10':
            im[0] -= 0.7203571
            im[0] /= 0.2807092
            im[1] -= 0.6721025
            im[1] /= 0.2561926
            im[2] -= 0.6199647
            im[2] /= 0.26200315
            im[3] -= 0.49367973
            im[3] /= 0.19068004
            im[4] -= 0.7766791
            im[4] /= 0.17894566
            im[5] -= 0.55923575
            im[5] /= 0.22804247
        elif self.normalize and self.dataset == 'modelnet40':
            im[0] -= 0.7139052
            im[0] /= 0.27971452
            im[1] -= 0.6935615
            im[1] /= 0.2606435
            im[2] -= 0.5850884
            im[2] /= 0.27366385
            im[3] -= 0.53355956
            im[3] /= 0.21440032
            im[4] -= 0.76255935
            im[4] /= 0.19869797
            im[5] -= 0.5651189
            im[5] /= 0.24401328
        im = im.astype(np.float32)  # pylint: disable=E1101

        return im

    def __repr__(self):
        return self.__class__.__name__ + '(level={0}, points={1})'.format(self.level, self.pts)


class CacheNPY:
    def __init__(self, prefix, transform, sp_mesh_dir, sp_mesh_level=5):
        self.transform = transform
        self.prefix = prefix

    def check_trans(self, file_path):
        print("transform {}...".format(file_path))
        try:
            return self.transform(file_path)
        except:
            print("Exception during transform of {}".format(file_path))
            raise

    def __call__(self, file_path):
        head, tail = os.path.split(file_path)
        root, _ = os.path.splitext(tail)
        npy_path = os.path.join(head, self.prefix + root + '.npy')

        exists = os.path.exists(npy_path)
        try:
            img = np.load(npy_path)
        except (OSError, FileNotFoundError):
            print(file_path)
            img = self.check_trans(file_path)
            np.save(npy_path, img)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(prefix={0}, transform={1})'.format(self.prefix, self.transform)


class ModelNet(torch.utils.data.Dataset):
    '''
    Process ModelNet(10/40) and output valid obj/off files content
    '''

    def __init__(self, root, dataset, partition, transform=None, target_transform=None, ftype='off'):
        self.root = os.path.expanduser(root)
        self.ftype = ftype
        if not dataset in ["modelnet10", "modelnet40"]:
            raise ValueError("Invalid dataset [modelnet10/modelnet40]")
        if not partition in ["train", "test"]:
            raise ValueError("Invalid partition [train/test]")

        self.dir = os.path.join(self.root, dataset + "_" + partition)
        self.transform = transform
        self.target_transform = target_transform

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        self.files = sorted(glob.glob(os.path.join(self.dir, '*.'+self.ftype)))
 
        self.labels = {}
        for fpath in self.files:
            fname = os.path.splitext(os.path.basename(fpath))[0]
            self.labels[fname] = "_".join(fname.split('_')[:-1]) # extract label. this is to deal with night_stand

    def __getitem__(self, index):
        img = f = self.files[index]

        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                print("                     Failing model: {}".format(f))

        i = os.path.splitext(os.path.basename(f))[0]
        target = self.labels[i]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.files)

    def _check_exists(self):
        files = glob.glob(os.path.join(self.dir, '*.'+self.ftype))

        return len(files) > 0