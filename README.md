# Cherab EDGE2D

Cherab add-on module for EDGE2D simulations.

This module enables the creation of Cherab plasma objects from EDGE2D simulations.
EDGE2D tran files are supported.
Please see the examples in the [demos](demos) directory for an illustration of how to use the module.

## Installation

It is recommended to install Cherab in a [virtual environment](https://docs.python.org/3/tutorial/venv.html).
This will enable installation of packages without modifying the system Python installation, which is particularly important on shared systems.
To create a virtual environment, do the following:

```bash
python3 -m venv ~/venvs/cherab-venv
```

After the virtual environment is created, it can be activated by running:

```bash
source ~/venvs/cherab-venv/bin/activate
```

### Users

This module depends on the core Cherab framework and Eproc EDGE2D processing library.

Cherab core, and all of its dependencies, are available on PyPI and can be installed using `pip`.
However, the EDGE2D module will need to be installed from this repository.

Note also that a [bug](https://github.com/cython/cython/issues/2918) in Cython prevents Cherab submodules from installing correctly.
This bug is fixed, but not yet released in the stable version of Cython.
As a result, you will need to install the latest alpha version of Cython before installing this package.

First, clone this repository, then do:

```bash
pip install -U cython==3.0a5
pip install cherab
pip install <path-to-cherab-edge2d>
```

This will pull in `cherab-core`, `raysect` `numpy` and other dependencies, then build and install the EDGE2D module.


Eproc can be obtained from [CCFE Gitlab repository](https://git.ccfe.ac.uk/jintrac/EPROC).
Note that Eproc is tied to JET computing clusters (Freia/Heimdall).

Numpy 1.17.4 is required to call Eproc functions from Python 3.7:

```bash
pip install --user numpy==1.17.4
```

When installed, update the .bashrc file to setup the environment:

```bash
EPROCDIR=~/eproc
export IDL_PATH=$EPROCDIR/idl/pro:$EPROCDIR/idl/prolib:$IDL_PATH
export IDL_PATH=$EPROCDIR/python/eproc:$EPROCDIR/python/eproc:$IDL_PATH
export LD_LIBRARY_PATH=$EPROCDIR/lib64:$EPROCDIR/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHONPATH:$EPROCDIR/python
export PYTHONPATH=$PYTHONPATH:$EPROCDIR/python/eproc
```


### Developers

For developing the code, it is recommended to use local checkouts of `cherab-core` and `raysect`, as well as `cherab-edge2d`.
Development should be done against the `development` branch of this repository, and any modifications submitted as pull requests to be merged back into `development`.

To install the package in develop mode, so that local changes are immediately visible without needing to reinstall, install with:

```
pip install -e <path-to-cherab-edge2d>
```

If you are modifying Cython files you will need to run `./dev/build.sh` from this directory in order to rebuild the extension modules.
They will then be used when Python is restarted.

