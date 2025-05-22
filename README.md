# Benchmarking Stochastic Approximation Algorithms for Fairness-Constrained Training of Deep Neural Networks

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository provides a tool to compare stochastic-constrained stochastic optimization algorithms on a _fair learning_ task.

## Table of Contents
1. [Basic installation instructions](#basic-installation-instructions)
2. [Reproducing the Benchmark](#reproducing-the-benchmark)
3. [Extending the benchmark](#extending-the-benchmark) <!-- 6. [Citing humancompatible/train](#Citing-humancompatible/train) -->
4. [License and terms of use](#license-and-terms-of-use)
5. [References](#references)

Humancompatible/train is still under active development! If you find bugs or have feature
requests, please file a
[Github issue](https://github.com/humancompatible/train/issues). 

## Basic installation instructions
The code requires Python version ```3.10```.

1. Create a virtual environment
```
python3.10 -m venv fairbenchenv
source fairbenchenv/bin/activate
```
2. Install from source.
```
git clone https://github.com/humancompatible/train.git
cd train
pip install -r requirements.txt
pip install .
```
<!-- Install via pip -->
<!-- ``` -->
<!-- pip install folktables -->
<!-- ``` -->

## Reproducing the Benchmark

### Running the algorithms

The benchmark comprises the following algorithms:
- Stochastic Ghost [[2]](#2),
- SSL-ALM [[3]](#3),
- Stochastic Switching Subgradient [[4]](#4).

To reproduce the experiments of the paper, run the following:
``` python
cd experiments
python run_folktables.py data=folktables alg=sslalm
python run_folktables.py data=folktables alg=alm
python run_folktables.py data=folktables alg=ghost
python run_folktables.py data=folktables alg=ssg
python run_folktables.py data=folktables alg=sgd     # baseline, no fairness
python run_folktables.py data=folktables alg=fairret # baseline, fairness with regularizer
```
Each command will start 10 runs of the `alg`, 30 seconds each.
The results will be saved to `experiments/utils/saved_models` and `experiments/utils/exp_results`.
<!-- In the repository, we include the configuration needed to reproduce the experiments in the paper. To do so, go to `experiments` and run `python run_folktables.py data=folktables alg=sslalm`. -->
<!-- Repeat for the other algorithms by changing the `alg` parameter. -->

This repository uses [Hydra](https://hydra.cc/) to manage parameters; see `experiments/conf` for configuration files. 
* To change the parameters of the experiment, such as the number of runs for each algorithm, run time, the dataset used (*note: for now supports only Folktables*) - use `experiment.yaml`. 
* To change the dataset settings - such as file location - or do dataset-specific adjustments, use `data/{dataset_name}.yaml`
* To change algorithm hyperparameters, use `alg/{algorithm_name}.yaml`.
* To change constraint hyperparameters, use `constraint/{constraint_name}.yaml`

<!-- ; it is installed as one of the dependencies. -->
<!-- To learn more about using Hydra, please check out the [official tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app). -->

### Producing plots
The plots and tables like the ones in the paper can be produced using the two notebooks. `experiments/algo_plots.ipynb` houses the convergence plots, and `experiments/model_plots.ipynb` - all the others.

## Extending the benchmark

**To add a new algorithm**, you can subclass the ```Algorithm``` class. Before you can run it, you will need to follow these steps:
1. In the `experiments/conf/alg` folder, add a `.yaml` file with `import_name: {ClassName}` (so the code knows which algorithm to import) and the desired keyword parameter values under `params`:

```
import_name: ClassName

params:
  param_name_1: value
  param_name_2: value
```

2. In `src/algorithms/__init__.py`, add `from .{filename} import {ClassName}` (so the code is able to import it).

Now you can run the algorithm by executing `python run_folktables.py data=folktables alg={yaml_file_name}`, or by changing the experiment config files.

**To add a different constraint formulation**, you can use the `FairnessConstraint` class by passing your callable function to the constructor as `fn`. If you use `run_folktables.py`, you can add a new constraint function by following the steps:

1. Add a `.yaml` file with `import_name: {FunctionName}`, along with the desired batch size and bound (*to be reworked for more generality*), to the `experiments/conf/constraint` folder
2. Import it in `src/constraints/__init__.py` as in step 2 above.

Now, to run the code with your constraint, use the `constraint` field in the main config.

## License and terms of use

Humancompatible/train is provided under the Apache 2.0 Licence.

The package relies on the Folktables package, provided under MIT Licence.
It provides code to download data from the American Community Survey
(ACS) Public Use Microdata Sample (PUMS) files managed by the US Census Bureau.
The data itself is governed by the terms of use provided by the Census Bureau.
For more information, see https://www.census.gov/data/developers/about/terms-of-service.html

<!-- ## Cite this work -->

<!-- If you use this work, we encourage you to cite our paper, and the folktables dataset [[1]](#1). -->

<!-- ``` -->
<!-- @article{ding2021retiring, -->
<!--   title={Retiring Adult: New Datasets for Fair Machine Learning}, -->
<!--   author={Ding, Frances and Hardt, Moritz and Miller, John and Schmidt, Ludwig}, -->
<!--   journal={Advances in Neural Information Processing Systems}, -->
<!--   volume={34}, -->
<!--   year={2021} -->
<!-- } -->
<!-- ``` -->

## Future work

- Add support for fairness constraints with >=2 subgroups (limitation of the code, not of the algorithms)
- Add support to datasets besides Folktables
- Move towards a more PyTorch-like API for optimizers

## References

<a id="1">[1]</a> 
Ding, Hardt & Miller et al. (2021) Retiring Adult: New Datasets for Fair Machine Learning, Curran Associates, Inc..

<a id="2">[2]</a> 
Facchinei & Kungurtsev (2023) Stochastic Approximation for Expectation Objective and Expectation Inequality-Constrained Nonconvex Optimization, arXiv.

<a id="3">[3]</a> 
Huang, Zhang & Alacaoglu (2025) Stochastic Smoothed Primal-Dual Algorithms for Nonconvex Optimization with Linear Inequality Constraints, arXiv.

<a id="4">[4]</a> 
Huang & Lin (2023) Oracle Complexity of Single-Loop Switching Subgradient Methods for Non-Smooth Weakly Convex Functional Constrained Optimization, Curran Associates Inc..

