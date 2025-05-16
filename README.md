# Benchmarking Stochastic Approximation Algorithms for Fairness-Constrained Training of Deep Neural Networks

This repository provides a tool to compare stochastic-constrained stochastic optimization algorithms on a _fair learning_ task.

## Dependencies

This code requires Python version ```3.10```. All dependencies are listed in the ```requirements.txt``` file and can be installed by running ```pip install -r requirements.txt```.

## Reproducibility

To reproduce the experiments in the paper, run ```experiments/run_folktables.py``` with the dataset name, algorithm name and hyperparameters as command line arguments, like below:

```run_folktables.py --algorithm sslalm --state OK --task income --constraint loss --loss_bound 0.005 --num_exp 10 --time 30 --batch_size 8 -mu 2. -rho 1. -tau 0.01 -eta 5e-2 -beta 0.5```

This will start 10 runs of the SSL-ALM algorithm, 30 seconds each, and save the model and the results in the ```experiments/utils/saved_models``` and ```experiments/utils/exp_results``` folders.

## Running your own experiments

To add a different constraint formulation, you can use the ```FairnessConstraint``` class by passing your callable function to the constructor as ```fn```.
To add a new algorithm, you can subclass the ```Algorithm``` class.

## Plots!

The plots and tables like the ones in the paper can be produced using the two notebooks. `experiments/algo_plots.ipynb` houses the convergence plots, and `experiments/model_plots.ipynb` - all the others.

**Warning**: As of 16/05, Folktables seems to be unable to connect to the American census servers. This means that downloading the dataset through the code is not possible. Manual download requires two files: the .csv dataset, at https://www2.census.gov/programs-surveys/acs/data/pums/`{year}`/`{horizon}`, and the corresponding .csv description, at https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/; use the flag ```--no-download```. By default, the files will be placed in `experiments/utils/raw_data/{task}/{year}/{horizon}` (e.g. `experiments/utils/raw_data/income/2018/1-Year/{filename}.csv`). A custom path can be specified with the --data_path argument, but it has to have the form `*/{year}/{horizon}/`.