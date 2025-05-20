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
```
<!-- Install via pip -->
<!-- ``` -->
<!-- pip install folktables -->
<!-- ``` -->

## Reproducing the Benchmark

### Running the algorithms
To reproduce the experiments in the paper, run ```experiments/run_folktables.py``` with the dataset name, algorithm name and hyperparameters as command line arguments, like below:
```run_folktables.py --algorithm sslalm --state OK --task income --constraint loss --loss_bound 0.005 --num_exp 10 --time 30 --batch_size 8 -mu 2. -rho 1. -tau 0.01 -eta 5e-2 -beta 0.5```
This will start 10 runs of the SSL-ALM algorithm, 30 seconds each, and save the model and the results in the ```experiments/utils/saved_models``` and ```experiments/utils/exp_results``` folders.

The benchmark comprises the following algorithms:
- Stochastic Ghost [[2]](#2),
- SSL-ALM [[3]](#3),
- Stochastic Switching Subgradient [[4]](#4).

### Producing plots
The plots and tables like the ones in the paper can be produced using the two notebooks. `experiments/algo_plots.ipynb` houses the convergence plots, and `experiments/model_plots.ipynb` - all the others.

**Warning**: As of 16/05, Folktables seems to be unable to connect to the American census servers. This means that downloading the dataset through the code is not possible. Manual download requires two files: the .csv dataset, at https://www2.census.gov/programs-surveys/acs/data/pums/`{year}`/`{horizon}`, and the corresponding .csv description, at https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/; use the flag ```--no-download```. By default, the files will be placed in `experiments/utils/raw_data/{task}/{year}/{horizon}` (e.g. `experiments/utils/raw_data/income/2018/1-Year/{filename}.csv`). A custom path can be specified with the --data_path argument; `folktables` will look for .csv files at `{custom_path}/{year}/{horizon}/`.

## Extending the benchmark 

To add a different constraint formulation, you can use the ```FairnessConstraint``` class by passing your callable function to the constructor as ```fn```.
To add a new algorithm, you can subclass the ```Algorithm``` class.

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


## References

<a id="1">[1]</a> 
Ding, Hardt & Miller et al. (2021) Retiring Adult: New Datasets for Fair Machine Learning, Curran Associates, Inc..

<a id="2">[2]</a> 
Facchinei & Kungurtsev (2023) Stochastic Approximation for Expectation Objective and Expectation Inequality-Constrained Nonconvex Optimization, arXiv.

<a id="3">[3]</a> 
Huang, Zhang & Alacaoglu (2025) Stochastic Smoothed Primal-Dual Algorithms for Nonconvex Optimization with Linear Inequality Constraints, arXiv.

<a id="4">[4]</a> 
Huang & Lin (2023) Oracle Complexity of Single-Loop Switching Subgradient Methods for Non-Smooth Weakly Convex Functional Constrained Optimization, Curran Associates Inc..

