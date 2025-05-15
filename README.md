# Benchmarking Stochastic Approximation Algorithms for Fairness-Constrained Training of Deep Neural Networks

This repository provides a tool to compare stochastic-constrained stochastic optimization algorithms on a _fair learning_ task.

## Dependencies

This code requires Python version ```3.10```. All dependencies are listed in the ```requirements.txt``` file and can be installed by running ```pip install -r requirements.txt```.

## Reproducibility

To reproduce the experiments in the paper, run ```experiments/run_folktables.py``` with the dataset name, algorithm name and hyperparameters as command line arguments, like below:

```run_folktables.py --algorithm 'sslalm' --state 'OK' --task 'income' --constraint 'loss' --loss_bound 0.005 --num_exp 10 --time 30 --batch_size 8 -mu 2. -rho 1. -tau 0.01 -eta 5e-2 -beta 0.5```

This will start 10 runs of the SSL-ALM algorithm, 30 seconds each, and save the model and the results in the ```experiments/utils/saved_models``` and ```experiments/utils/exp_results``` folders.

## Running your own experiments

To add a different constraint formulation, you can use the ```FairnessConstraint``` class by passing your callable function to the constructor as ```fn```.
To add a new algorithm, you can subclass the ```Algorithm``` class.

## Plots!

The plots and tables like the ones in the paper can be produced using the two notebooks. `experiments/algo_plots.ipynb` houses the convergence plots, and `experiments/model_plots.ipynb` - all the others.
