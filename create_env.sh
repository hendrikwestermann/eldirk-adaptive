rm -rf env
python3 -m venv env

source ./env/bin/activate

pip3 install numpy==1.25.2
pip3 install numba==0.58.0
pip3 install pyvista==0.40.0
pip3 install scipy==1.11.3
pip3 install meshio==5.3.0
pip3 install pypardiso==0.4.2
pip3 install llvmlite==0.41.0
