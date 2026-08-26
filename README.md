[![PyPI Downloads](https://static.pepy.tech/personalized-badge/probabilisticnn?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/probabilisticnn)
# ProbabilisticNN

`ProbabilisticNN` is a Python library for Probabilistic Neural Networks and General Regression Neural Networks with an sklearn-like interface.

At the current stage, the library is published on TestPyPI and supports:

- `PNN` for classification;
- `AdaptivePNN` with kernel bandwidth parameter tuning;
- `GRNN` for regression;
- `AdaptiveGRNN` with kernel bandwidth optimization;
- execution using the standard `numpy` implementation;
- accelerated inference via `numba`.

## Main Features

- sklearn-like interface: `fit`, `predict`, `predict_proba`;
- several kernel functions:
  - `gaussian`;
  - `laplacian`;
  - `exponential`;
- several kernel bandwidth parameterization schemes for `AdaptivePNN`:
  - `per_feature`;
  - `per_class`;
  - `per_class_per_feature`;
- accelerated `numba` backend for the computationally heavy part of inference;
- testing via `pytest` and `tox`.

## Requirements

- Python `3.12+`
- `numpy`
- `scipy`
- `scikit-learn`

For the accelerated backend, `numba` is additionally required.

## Installation

### Installation from TestPyPI

For regular usage without `numba`:

```bash
python -m pip install ProbabilisticNN
```

For installation with the `numba` backend:

```bash
python -m pip install "ProbabilisticNN[numba]"
```

For development and testing:

```bash
python -m pip install "ProbabilisticNN[dev]"
```

### Local Installation from the Repository

This requires `setuptools`.

```bash
python -m build
python -m pip install -e ".[dev]"
```

## Quick Start

### Classification with `PNN`

```python
import numpy as np

from probabilisticnn.pnn import PNN

X_train = np.array([
    [0.0, 0.0],
    [0.0, 0.2],
    [1.0, 1.0],
    [1.0, 1.2],
])
y_train = np.array(["class_a", "class_a", "class_b", "class_b"])

X_test = np.array([
    [0.0, 0.1],
    [1.0, 1.1],
])

model = PNN(
    bandwidth=0.25,
    kernel="gaussian",
    normalize=False,
)

model.fit(X_train, y_train)

labels = model.predict(X_test)
proba = model.predict_proba(X_test)

print(labels)
print(proba)
```

### Classification with `AdaptivePNN`

```python
import numpy as np

from probabilisticnn.pnn import AdaptivePNN

X_train = np.array([
    [0.0, 0.0],
    [0.0, 0.2],
    [1.0, 1.0],
    [1.0, 1.2],
])
y_train = np.array([0, 0, 1, 1])

model = AdaptivePNN(
    kernel="gaussian",
    bandwidth_sharing="per_feature",
    loss="correct_class_probability",
    solver="auto",
    normalize=False,
)

model.fit(X_train, y_train)

print(model.predict(X_train))
print(model.predict_proba(X_train))
print(model.bandwidth_)
```

### Regression with `GRNN`

```python
import numpy as np

from probabilisticnn.grnn import GRNN

X_train = np.array([
    [0.0, 0.0],
    [0.0, 0.2],
    [1.0, 1.0],
    [1.0, 1.2],
])
y_train = np.array([1.0, 1.1, 3.0, 3.1])

X_test = np.array([
    [0.0, 0.1],
    [1.0, 1.1],
])

model = GRNN(
    bandwidth=0.5,
    kernel="gaussian",
    normalize=False,
)

model.fit(X_train, y_train)
pred = model.predict(X_test)

print(pred)
```

### Regression with `AdaptiveGRNN`

```python
import numpy as np

from probabilisticnn.grnn import AdaptiveGRNN

X_train = np.array([
    [0.0, 0.0],
    [0.0, 0.2],
    [1.0, 1.0],
    [1.0, 1.2],
])
y_train = np.array([1.0, 1.1, 3.0, 3.1])

model = AdaptiveGRNN(
    kernel="gaussian",
    loss="mse",
    solver="auto",
    normalize=False,
)

model.fit(X_train, y_train)

print(model.predict(X_train))
print(model.bandwidth_)
```

## Acceleration with `numba`

For `PNN`, `AdaptivePNN`, `GRNN`, and `AdaptiveGRNN`, you can choose:

- `backend="numpy"` — the main option;
- `backend="numba"` — the accelerated option.

Example:

```python
from probabilisticnn.pnn import PNN

model = PNN(
    bandwidth=0.25,
    kernel="gaussian",
    backend="numba",
    normalize=False,
)
```

Important:

- the first `predict` call for the `numba` branch may be slower because of JIT compilation;
- the main speed gain is achieved during the computation of kernel function values;
- if `numba` is not installed, the `numba` backend will be unavailable.

## Public Interface

Models should be imported like this:

```python
from probabilisticnn.pnn import PNN, AdaptivePNN
from probabilisticnn.grnn import GRNN, AdaptiveGRNN
```

## Model Parameters

### `PNN`

Main parameters:

- `bandwidth` — fixed kernel bandwidth;
- `kernel` — kernel type;
- `losses` — class weighting scheme / misclassification cost for classes;
- `normalize` — whether to use L2 normalization of inputs;
- `backend` — `numpy` or `numba`;
- `compute_dtype` — `auto`, `float32`, `float64`.

### `AdaptivePNN`

Additionally supports:

- `loss` — optimization objective function;
- `bandwidth_sharing` — method for defining bandwidth parameters;
- `max_iter` — maximum number of optimization steps;
- `solver` — optimization method;
- `solver_options` — additional optimizer parameters.

### `GRNN`

Main parameters:

- `bandwidth`;
- `kernel`;
- `backend`;
- `compute_dtype`;
- `normalize`.

### `AdaptiveGRNN`

Additionally supports:

- `loss` — loss function for regression;
- `max_iter`;
- `solver`;
- `solver_options`;
- `normalize`.

`AdaptiveGRNN` optimizes the kernel bandwidth using the `per_feature` scheme.

## Testing

Preferred way to run tests:

```bash
tox
```

Or run a specific test file:

```bash
pytest tests/test_pnn.py
```

Since the tests import the installed package, it is most convenient to run them in a prepared environment after installing the dependencies.

## Performance

The repository contains separate materials for measuring performance:

- [benchmarks/inference_bemchmarks.py](benchmarks/inference_bemchmarks.py)
- [benchmarks/benchmarking.ipynb](benchmarks/benchmarking.ipynb)

## Examples

An additional usage example is located at:

- [examples/basic_usage.ipynb](examples/basic_usage.ipynb)

## License

The project is distributed under the MIT license. See the [LICENCE](LICENCE) file.
