from mesh import icosphere
import pyigl as igl
from iglhelpers import *
import pickle
from scipy.io import loadmat


def test1():
    sp = icosphere(7)
    print(sp.nv, sp.nf)
    # V = p2e(sp.vertices)
    # F = p2e(sp.faces)
    # viewer = igl.glfw.Viewer()
    # viewer.data().set_mesh(V, F)
    # # viewer.data().show_lines = False
    # viewer.launch()
    # print(sp.vertices.shape)
    # print(sp.faces.shape)


def test2(field_name):
    p = pickle.load(open("icosphere_7.pkl", "rb"))
    u = np.load("spdata.npz")["data"]
    l = np.load("spdata.npz")["labels"]
    v = p['V']
    f = p['F']
    u = np.array(u[:, 0])
    # Take gradient x and y
    gu = np.reshape(p['G'].dot(u), [-1, 3], order='F')
    ux = np.sum(gu * p['EW'], axis=-1)
    uy = np.sum(gu * p['NS'], axis=-1)
    ux_vertex = p['F2V'] * ux
    uy_vertex = p['F2V'] * uy
    lap = p['L'].dot(u)
    if field_name == "I" or field_name == "i":
        Q = p2e(u)
    elif field_name == "DX" or field_name == "dx":
        Q = p2e(ux_vertex)
    elif field_name == "DY" or field_name == "dy":
        Q = p2e(uy_vertex)
    elif field_name == "L" or field_name == "lap":
        Q = p2e(lap)
    elif field_name == "labels":
        Q = p2e(l)
    else:
        print("[!] Incorrect field name.")
        assert(0)
    V = p2e(v)
    F = p2e(f)
    C = igl.eigen.MatrixXd()
    igl.colormap(igl.COLOR_MAP_TYPE_VIRIDIS, Q, True, C)
    viewer = igl.glfw.Viewer()
    viewer.data().clear()
    viewer.data().set_mesh(V, F)
    viewer.data().set_colors(C)
    viewer.data().show_lines = False
    bc = viewer.core.background_color
    bc.setConstant(.5)
    viewer.core.background_color = bc
    viewer.launch()


if __name__ == '__main__':
    test2("DX")