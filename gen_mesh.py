import sys
import os

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
        for i in range(8):
            url = 'http://island.me.berkeley.edu/ugscnn/mesh_files/' + fname.format(i)
            command = ["wget", "--no-check-certificate", "-P", dest, url]
            try:
                download_state = subprocess.call(command)
            except Exception as e:
                print(e)

if __name__ == '__main__':
    main()
