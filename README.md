# train

Fairness-constrained training of machine learning models in TensorFlow, PyTorch, and beyond

This module provides a framework for testing and enforcing fairness in machine learning algorithms using Stochastic Approximation techniques. It serves as a benchmark for evaluating the performance of various algorithms under fairness constraints.   

Features:

Implements Empirical Risk Minimization (ERM) with fairness constraints.
Supports multiple fairness constraints such as demographic parity, equalized odds, and others.
Utilizes Stochastic Approximation techniques to efficiently handle large datasets and complex models.
Provides tools for benchmarking the fairness and performance of different algorithms.


Requirements:

Python :

Python 3.10

Libraries:

numpy
scipy
qpsolvers
autoray
pytorch
tensorflow
StochasticGhost


Running the benchmark:

python income.py --model "{backend name}" --optimizer "{optimizer name}"

Help:

python income.py --help 

model values: pytorch_connect, tensorflow_connect
optimizer values: StochasticGhost

