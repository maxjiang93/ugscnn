from mesh_utils import *
import scipy.sparse as sparse
import pyigl as igl
import pickle


class icosphere(object):
    def __init__(self, level=0):
        self.level = level
        self.vertices, self.faces = self.icosahedron()
        self.nv_prev = self.vertices.shape[0]
        for l in range(self.level):
            self.nv_prev = self.vertices.shape[0]
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
                     "F2V": self.F2V}

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
        mid = np.vstack([triangles[:, g, :].mean(axis=1) for g in [[0, 1],
                                                                   [1, 2],
                                                                   [2, 0]]])
        mid_idx = (np.arange(len(face_index) * 3)).reshape((3, -1)).T
        # for adjacent faces we are going to be generating the same midpoint
        # twice, so we handle it here by finding the unique vertices
        unique, inverse = unique_rows(mid)

        mid = mid[unique]
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

        self.vertices = new_vertices
        self.faces = new_faces
    
    def normalize(self, radius=1):
        '''
        Reproject to spherical surface
        '''
        vectors = self.vertices
        scalar = (vectors ** 2).sum(axis=1)**.5
        unit = vectors / scalar.reshape((-1, 1))
        offset = radius - scalar
        self.vertices += unit * offset.reshape((-1, 1))
        
    def icosahedron(self):
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
        return vertices, faces

    def xyz2latlong(self):
        x, y, z = self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2]
        long = np.arctan2(y, x)
        xy2 = x**2 + y**2
        lat = np.arctan2(z, np.sqrt(xy2))
        return lat, long

    def construct_matrices(self):
        """
        Construct FEM matrices
        """
        V = p2e(self.vertices)
        F = p2e(self.faces)
        # Compute gradient operator: #F*3 by #V
        G = igl.eigen.SparseMatrixd()
        L = igl.eigen.SparseMatrixd()
        N = igl.eigen.MatrixXd()
        A = igl.eigen.MatrixXd()
        igl.grad(V, F, G)
        igl.cotmatrix(V, F, L)
        igl.per_face_normals(V, F, N)
        igl.doublearea(V, F, A)
        G = e2p(G)
        L = e2p(L)
        N = e2p(N)
        A = e2p(A)
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

        # Compute vertex mean matrix
        self.G = G  # gradient matrix
        self.L = L  # laplacian matrix
        self.N = N  # normal vectors (per-triangle)
        self.NS = NS  # north-south vectors (per-triangle)
        self.EW = EW  # east-west vectors (per-triangle)
        self.F2V = F2V  # map face quantities to vertices

    def export_mesh_info(self, filename):
        """Write mesh info as pickle file"""
        with open(filename, "wb") as f:
            pickle.dump(self.info, f)
