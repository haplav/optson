# Optson: A Flexible Optimization Toolbox

[![pipeline](https://gitlab.com/swp_ethz/optson/badges/main/pipeline.svg)]()
[![coverage](https://gitlab.com/swp_ethz/optson/badges/main/coverage.svg)]()

## Summary

* Optson is python toolbox for nonlinear unconstrained optimization.
* It offers Adam, Trust-Region L-BGFS and Steepest Descent methods and more can be added easily.
* It supports both "batch" and "mini-batch" approaches via specific implementations of the `BatchManager` class.
* A user can define a sublass of the `Problem` class and provide implementations of `f()` (misfit) and `g()` (gradient).
* An instance of this class as well as an initial model are then passed to Optson, after which Optson will
perform optimization according to the specified settings.

Full documentation: https://swp_ethz.gitlab.io/optson

## Installation
* Make sure you have an environment with Python version 3.9 or higher.  
  However, you can use our Conda environment files for that, mentioned below.
* Clone this repository and change to the resulting directory.
* To install just minimal dependencies run:
  ```
  # skip if you don't use Conda
  conda env create -f conda-env-basic.yml
  conda activate optson
  ```
  ```
  pip install -e "."
  ```
* To install also dependencies needed for tutorials, use the `tutorials` dependencies:
  ```
  # skip if you don't use Conda
  conda env create -f conda-env-tutorials.yml
  conda activate optson
  ```
  ```
  pip install -e ".[tutorials]"
  ```


### Optional support for JAX and PyTorch
If you would like to use JAX Arrays, or PyTorch Tensors, you can install Optson with JAX and/or PyTorch support.

* To install Optson with JAX support, add the `jax` dependency group, e.g.:
  ```
  # skip if you don't use Conda
  conda env create -f conda-env-tutorials.yml
  conda activate optson
  conda install jax
  ```
  ```
  pip install -e ".[tutorials,jax]"
  ```
* To install Optson with PyTorch support, add the `torch` dependency group, e.g.:
  ```
  # skip if you don't use Conda
  conda env create -f conda-env-tutorials.yml
  conda activate optson
  conda install torch
  ```
  ```
  pip install -e ".[tutorials,torch]"
  ```

### Testing
If you want to run the tests, ensure you have the required dependencies.
To do so, run:
```
pip install -v -e ".[test]"
```

To run the tests, cd into the toplevel ``optson`` directory and execute:
```
py.test
```

This should ensure that your installation is working and optson is working as intended.


## Try it out
Please have a look at `tutorials/` for example usage.


## Updating Optson

To update Optson, change into the top-level `optson` directory and type
```
git pull
pip install -v -e .
```


## Contributing to Optson

If you would like to contribute, feel free to make a merge request with
your suggested feature.

Before incorporation, the merge request has to pass the CI testing routines and
be reviewed by one of the maintainers, Vaclav or Dirk-Philip.

To install everything we use in our CI, including tools for docs build, do:
```
# skip if you don't use Conda
conda env create -f ci/conda-env.yml
conda activate optson
```
```
pip install -e ".[test,docs]"
```

You can then execute the test script ``optson/ci/test.sh`` locally.
We encourage you to do this before creating your MR.
