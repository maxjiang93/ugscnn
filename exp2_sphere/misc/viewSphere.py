import pyigl as igl
from scipy.io import loadmat
import numpy as np
from iglhelpers import p2e, e2p

def main():
	data = loadmat("spdata.mat")
	x, y, z = data['x'], data['y'], data['z']
	azi, ele = np.squeeze(data['azi']), np.squeeze(data['ele'])
	v = np.concatenate((x, y, z), axis=0).T
	f = data['f']

	u = data['data']
	U = p2e(u[..., 0])
	Lb = p2e(np.squeeze(data['labels']))
	V = p2e(v)
	F = p2e(f)


	Ele = p2e(ele)
	G = igl.eigen.SparseMatrixd()
	igl.grad(V, F, G)
	NS = (G * Ele).MapMatrix(F.rows(), 3)
	NS_mag = NS.rowwiseNorm()
	# compute normals (per-face)
	N = igl.eigen.MatrixXd()
	igl.per_face_normals(V, F, N)
	EW = p2e(np.cross(e2p(NS), e2p(N)))
	EW_mag = EW.rowwiseNorm()

	# set color

	C = igl.eigen.MatrixXd()
	D = (G * U).MapMatrix(F.rows(), 3)
	Dx = D.cwiseProduct(EW).rowwiseSum()
	Dy = D.cwiseProduct(NS).rowwiseSum()
	# igl.jet(U, True, C)
	igl.colormap(igl.COLOR_MAP_TYPE_VIRIDIS, U, True, C)
	# C.setConstant(V.rows(), 3, .9)

	# plot vector lines
	# Average edge length divided by average gradient (for scaling)
	max_size_ns = igl.avg_edge_length(V, F) / NS_mag.mean()
	max_size_ew = igl.avg_edge_length(V, F) / EW_mag.mean()
	# Draw a black segment in direction of gradient at face barycenters
	BC = igl.eigen.MatrixXd()
	igl.barycenter(V, F, BC)
	blue = igl.eigen.MatrixXd([[0.0, 0.0, 0.5]])
	red = igl.eigen.MatrixXd([[0.5, 0.0, 0.0]])


	viewer = igl.glfw.Viewer()
	viewer.data().set_mesh(V, F)
	viewer.data().set_colors(C)
	# viewer.data().add_edges(BC, BC + max_size_ns * NS, blue)
	# viewer.data().add_edges(BC, BC + max_size_ew * EW, red)
	viewer.data().show_lines = False
	viewer.launch()


if __name__ == '__main__':
    main()