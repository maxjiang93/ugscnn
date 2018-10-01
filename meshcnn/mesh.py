from .mesh_utils import *
import scipy.sparse as sparse
import pyigl as igl
import pickle
import os


def export_spheres(int_list, dest_folder):
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)
    fname = os.path.join(dest_folder, "icosphere_{}.pkl")
    for i in int_list:
        s = icosphere(i)
        s.export_mesh_info(fname.format(i))


class icosphere(object):
    def __init__(self, level=0, upward=False):
        self.level = level
        self.vertices, self.faces = self.icosahedron(upward=upward)
        self.intp = None
        self.v0, self.f0 = self.vertices.copy(), self.faces.copy()
        for l in range(self.level):
            self.subdivide()
            self.normalize()
        self.lat, self.long = self.xyz2latlong()
        self.nf, self.nv = self.faces.shape[0], self.vertices.shape[0]
        self.nf = 20 * (4 ** self.level)
        self.ne = 30 * (4 ** self.level)
        self.nv = self.ne - self.nf + 2
        self.nv_prev = int((self.ne / 4) - (self.nf / 4) + 2)
        self.nv_next = int((self.ne * 4) - (self.nf * 4) + 2)

        self.construct_matrices()
        self.info = {"V": self.vertices,
                     "F": self.faces,
                     "nv_prev": self.nv_prev,
                     "nv_next": self.nv_next,
                     "G": self.G,
                     "L": self.L,
                     "N": self.N,
                     "NS": self.NS,
                     "EW": self.EW,
                     "F2V": self.F2V,
                     "M": self.M,
                     "Seq": self.Seq,
                     "Intp": self.Intp}

    def subdivide(self):
        """
        Subdivide a mesh into smaller triangles.
        """
        faces = self.faces
        vertices = self.vertices
        face_index = np.arange(len(faces))

        # the (c,3) int set of vertex indices
        faces = faces[face_index]
        # the (c, 3, 3) float set of points in the triangles
        triangles = vertices[faces]
        # the 3 midpoints of each triangle edge vstacked to a (3*c, 3) float
        src_idx = np.vstack([faces[:, g] for g in [[0, 1], [1, 2], [2, 0]]])
        mid = np.vstack([triangles[:, g, :].mean(axis=1) for g in [[0, 1],
                                                                   [1, 2],
                                                                   [2, 0]]])
        mid_idx = (np.arange(len(face_index) * 3)).reshape((3, -1)).T
        # for adjacent faces we are going to be generating the same midpoint
        # twice, so we handle it here by finding the unique vertices
        unique, inverse = unique_rows(mid)

        mid = mid[unique]
        src_idx = src_idx[unique]
        mid_idx = inverse[mid_idx] + len(vertices)
        # the new faces, with correct winding
        f = np.column_stack([faces[:, 0], mid_idx[:, 0], mid_idx[:, 2],
                             mid_idx[:, 0], faces[:, 1], mid_idx[:, 1],
                             mid_idx[:, 2], mid_idx[:, 1], faces[:, 2],
                             mid_idx[:, 0], mid_idx[:, 1], mid_idx[:, 2], ]).reshape((-1, 3))
        # add the 3 new faces per old face
        new_faces = np.vstack((faces, f[len(face_index):]))
        # replace the old face with a smaller face
        new_faces[face_index] = f[:len(face_index)]

        new_vertices = np.vstack((vertices, mid))
        # source ids
        nv = vertices.shape[0]
        identity_map = np.stack((np.arange(nv), np.arange(nv)), axis=1)
        src_id = np.concatenate((identity_map, src_idx), axis=0)

        self.vertices = new_vertices
        self.faces = new_faces
        self.intp = src_id
    
    def normalize(self, radius=1):
        '''
        Reproject to spherical surface
        '''
        vectors = self.vertices
        scalar = (vectors ** 2).sum(axis=1)**.5
        unit = vectors / scalar.reshape((-1, 1))
        offset = radius - scalar
        self.vertices += unit * offset.reshape((-1, 1))
        
    def icosahedron(self, upward=True):
        """
        Create an icosahedron, a 20 faced polyhedron.
        """
        t = (1.0 + 5.0**.5) / 2.0
        vertices = [-1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0, 0, -1, t, 0, 1, t,
                    0, -1, -t, 0, 1, -t, t, 0, -1, t, 0, 1, -t, 0, -1, -t, 0, 1]
        faces = [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
                 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
                 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
                 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1]
        # make every vertex have radius 1.0
        vertices = np.reshape(vertices, (-1, 3)) / 1.9021130325903071
        faces = np.reshape(faces, (-1, 3))
        if upward:
            vertices = self._upward(vertices, faces)
        return vertices, faces

    def xyz2latlong(self):
        x, y, z = self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2]
        long = np.arctan2(y, x)
        xy2 = x**2 + y**2
        lat = np.arctan2(z, np.sqrt(xy2))
        return lat, long

    def _upward(self, V_ico, F_ico, ind=11):
        V0 = V_ico[ind]
        Z0 = np.array([0, 0, 1])
        k = np.cross(V0, Z0)
        ct = np.dot(V0, Z0)
        st = -np.linalg.norm(k)
        R = self._rot_matrix(k, ct, st)
        V_ico = V_ico.dot(R)
        # rotate a neighbor to align with (+y)
        ni = self._find_neighbor(F_ico, ind)[0]
        vec = V_ico[ni].copy()
        vec[2] = 0
        vec = vec/np.linalg.norm(vec)
        y_ = np.eye(3)[1]

        k = np.eye(3)[2]
        crs = np.cross(vec, y_)
        ct = -np.dot(vec, y_)
        st = -np.sign(crs[-1])*np.linalg.norm(crs)
        R2 = self._rot_matrix(k, ct, st)
        V_ico = V_ico.dot(R2)
        return V_ico

    def _find_neighbor(self, F, ind):
        """find a icosahedron neighbor of vertex i"""
        FF = [F[i] for i in range(F.shape[0]) if ind in F[i]]
        FF = np.concatenate(FF)
        FF = np.unique(FF)
        neigh = [f for f in FF if f != ind]
        return neigh

    def _rot_matrix(self, rot_axis, cos_t, sin_t):
        k = rot_axis / np.linalg.norm(rot_axis)
        I = np.eye(3)

        R = []
        for i in range(3):
            v = I[i]
            vr = v*cos_t+np.cross(k, v)*sin_t+k*(k.dot(v))*(1-cos_t)
            R.append(vr)
        R = np.stack(R, axis=-1)
        return R

    def _ico_rot_matrix(self, ind):
        """
        return rotation matrix to perform permutation corresponding to 
        moving a certain icosahedron node to the top
        """
        v0_ = self.v0.copy()
        f0_ = self.f0.copy()
        V0 = v0_[ind]
        Z0 = np.array([0, 0, 1])

        # rotate the point to the top (+z)
        k = np.cross(V0, Z0)
        ct = np.dot(V0, Z0)
        st = -np.linalg.norm(k)
        R = self._rot_matrix(k, ct, st)
        v0_ = v0_.dot(R)

        # rotate a neighbor to align with (+y)
        ni = self._find_neighbor(f0_, ind)[0]
        vec = v0_[ni].copy()
        vec[2] = 0
        vec = vec/np.linalg.norm(vec)
        y_ = np.eye(3)[1]

        k = np.eye(3)[2]
        crs = np.cross(vec, y_)
        ct = np.dot(vec, y_)
        st = -np.sign(crs[-1])*np.linalg.norm(crs)

        R2 = self._rot_matrix(k, ct, st)
        return R.dot(R2)

    def _rotseq(self, V, acc=9):
        """sequence to move an original node on icosahedron to top"""
        seq = []
        for i in range(11):
            Vr = V.dot(self._ico_rot_matrix(i))
            # lexsort
            s1 = np.lexsort(np.round(V.T, acc))
            s2 = np.lexsort(np.round(Vr.T, acc))
            s = s1[np.argsort(s2)]
            seq.append(s)
        return tuple(seq)


    def construct_matrices(self):
        """
        Construct FEM matrices
        """
        V = p2e(self.vertices)
        F = p2e(self.faces)
        # Compute gradient operator: #F*3 by #V
        G = igl.eigen.SparseMatrixd()
        L = igl.eigen.SparseMatrixd()
        M = igl.eigen.SparseMatrixd()
        N = igl.eigen.MatrixXd()
        A = igl.eigen.MatrixXd()
        igl.grad(V, F, G)
        igl.cotmatrix(V, F, L)
        igl.per_face_normals(V, F, N)
        igl.doublearea(V, F, A)
        igl.massmatrix(V, F, igl.MASSMATRIX_TYPE_VORONOI, M)
        G = e2p(G)
        L = e2p(L)
        N = e2p(N)
        A = e2p(A)
        M = e2p(M)
        M = M.data
        # Compute latitude and longitude directional vector fields
        NS = np.reshape(G.dot(self.lat), [self.nf, 3], order='F')
        EW = np.cross(NS, N)
        # Compute F2V matrix (weigh by area)
        # adjacency
        i = self.faces.ravel()
        j = np.arange(self.nf).repeat(3)
        one = np.ones(self.nf * 3)
        adj = sparse.csc_matrix((one, (i, j)), shape=(self.nv, self.nf))
        tot_area = adj.dot(A)
        norm_area = A.ravel().repeat(3)/np.squeeze(tot_area[i])
        F2V = sparse.csc_matrix((norm_area, (i, j)), shape=(self.nv, self.nf))
        # Compute interpolation matrix
        if self.level > 0:
            intp = self.intp[self.nv_prev:]
            i = np.concatenate((np.arange(self.nv), np.arange(self.nv_prev, self.nv)))
            j = np.concatenate((np.arange(self.nv_prev), intp[:, 0], intp[:, 1]))
            ratio = np.concatenate((np.ones(self.nv_prev), 0.5*np.ones(2*intp.shape[0])))
            intp = sparse.csc_matrix((ratio, (i, j)), shape=(self.nv, self.nv_prev))
        else:
            intp = sparse.csc_matrix(np.eye(self.nv))
        

        # Compute vertex mean matrix
        self.G = G  # gradient matrix
        self.L = L  # laplacian matrix
        self.N = N  # normal vectors (per-triangle)
        self.NS = NS  # north-south vectors (per-triangle)
        self.EW = EW  # east-west vectors (per-triangle)
        self.F2V = F2V  # map face quantities to vertices
        self.M = M # mass matrix (area of voronoi cell around node. for integration)
        self.Seq = self._rotseq(self.vertices)
        self.Intp = intp

    def export_mesh_info(self, filename):
        """Write mesh info as pickle file"""
        with open(filename, "wb") as f:
            pickle.dump(self.info, f)

# from pdb import set_trace; set_trace()
# s = icosphere(2)
# a = 0
# import pyigl as igl
# from iglhelpers import p2e
# # visualize
# v = p2e(s.vertices)
# f = p2e(s.faces)
# viewer = igl.glfw.Viewer()
# viewer.data().set_mesh(v, f)
# viewer.launch()
