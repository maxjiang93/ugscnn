import sys
import os

import gdown


def main():
    try:
        sys.path.append("meshcnn")
        from mesh import export_spheres
        export_spheres(range(8), "mesh_files")
    except ImportError:
        print("ImportError occurred. Will download precomputed mesh files instead...")
        import subprocess
        dest = "mesh_files"
        if not os.path.exists(dest):
            os.makedirs(dest)
        fname = 'icosphere_{}.pkl'
        url = 'https://drive.google.com/uc?id=17RermILq8jGu1Oz2LgX-k98FnfzAzOW2'
        output = "mesh_files.zip"
        try:
            gdown.download(url, output, quiet=False)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    main()
