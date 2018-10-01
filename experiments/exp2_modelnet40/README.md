## 3D Shape classification on ModelNet40
### Dependencies
The code below has the following dependencies that can be installed by conda and pip.
```bash
conda install -c conda-forge rtree shapely  
conda install -c conda-forge pyembree  
pip install "trimesh[easy]"  
```
### Instruction
To run the experiment, execute the run script:
```bash
chmod +x run.sh
./run
```
The script will automatically start downloading the data files if it does not already exist.

We acknowledge code in this directory borrowed liberally from [S2CNN](https://github.com/jonas-koehler/s2cnn).
