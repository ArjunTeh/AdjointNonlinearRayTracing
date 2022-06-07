# AdjointEikonalTracerPrivate

## Installation Instructions

The code has been built and tested on Ubuntu 20.04

The project uses Python 3.8 and the following packages:
- `numpy`
- `pytorch >1.8 (with cuda)`
- `matplotlib`
- `tqdm`
- `PIL`

For the C++ part of the project, cmake is used, which also requires that all of the submodules are downloaded as well. This project relies on enoki.

Run the following command to generate the build files.
```bash
mkdir -p build
cd build

cmake .. \
    -DENOKI_CUDA=1 \
    -DENOKI_AUTODIFF=1 \
    -DENOKI_PYTHON=1
```

If you are using the conda environment, it might be necessary to directly link to your python executable and library:

```bash
cmake .. \
    -DENOKI_CUDA=1 \
    -DENOKI_AUTODIFF=1 \
    -DENOKI_PYTHON=1 \
    -DPYTHON_EXECUTABLE:FILEPATH=$CONDA_PYTHON_EXE \
    -DPYTHON_LIBRARY:FILEPATH=$CONDA_PREFIX/{path_to_python_library}
```

After the build files are generated, run:
```bash 
make
cd ..
source setpath.sh
```

This will run the build as well as source the output folders so that python can find the drrt library.

## Running the code
To run some of the experiments from the code, directly run one of the experiement scripts in the `core` folder.

```bash
python core/luneburg_opt.py
```

Otherwise, if you would like to use the code directly in your own python scripts, just import the drrt package and the enoki package.

```python
import drrt
import enoki

# your code here
```
