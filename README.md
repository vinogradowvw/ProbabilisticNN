# ProbabilisticNN

`ProbabilisticNN` — Python-библиотека для Probabilistic Neural Networks и General Regression Neural Networks со sklearn-подобным интерфейсом.

На текущем этапе библиотека опубликована на TestPyPI и поддерживает:

- `PNN` для классификации;
- `AdaptivePNN` с подбором параметров ширины ядра;
- `GRNN` для регрессии;
- `AdaptiveGRNN` с оптимизацией ширины ядра;
- выполнение на обычной реализации `numpy`;
- ускоренный вывод результата через `numba`.

## Основные возможности

- sklearn-подобный интерфейс: `fit`, `predict`, `predict_proba`;
- несколько ядерных функций:
  - `gaussian`;
  - `laplacian`;
  - `exponential`;
- несколько схем параметризации ширины ядра для `AdaptivePNN`:
  - `per_feature`;
  - `per_class`;
  - `per_class_per_feature`;
- ускоренный backend `numba` для вычислительно тяжелой части вывода результата;
- тестирование через `pytest` и `tox`.

## Требования

- Python `3.12+`
- `numpy`
- `scipy`
- `scikit-learn`

Для ускоренного backend дополнительно нужна `numba`.

## Установка

### Установка из TestPyPI

Для обычного использования без `numba`:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "ProbabilisticNN"
```

Для установки с `numba` backend:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "ProbabilisticNN[numba]"
```

Для разработки и тестирования:

```bash
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "ProbabilisticNN[dev]"
```

### Локальная установка из репозитория
Для это нужны setuptools.

```bash
python -m build
python -m pip install -e ".[dev]"
```

## Быстрый старт

### Классификация с `PNN`

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

### Классификация с `AdaptivePNN`

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

### Регрессия с `GRNN`

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

### Регрессия с `AdaptiveGRNN`

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

## Ускорение через `numba`

Для `PNN`, `AdaptivePNN`, `GRNN` и `AdaptiveGRNN` можно выбрать:

- `backend="numpy"` — основной вариант;
- `backend="numba"` — ускоренный вариант.

Пример:

```python
from probabilisticnn.pnn import PNN

model = PNN(
    bandwidth=0.25,
    kernel="gaussian",
    backend="numba",
    normalize=False,
)
```

Важно:

- первый вызов `predict` для `numba`-ветки может быть медленнее из-за JIT-компиляции;
- основной выигрыш по скорости достигается на этапе вычисления значений ядерных функций;
- если `numba` не установлена, backend `numba` будет недоступен.

## Публичный интерфейс

Импортировать модели следует так:

```python
from probabilisticnn.pnn import PNN, AdaptivePNN
from probabilisticnn.grnn import GRNN, AdaptiveGRNN
```

## Параметры моделей

### `PNN`

Основные параметры:

- `bandwidth` — фиксированная ширина ядра;
- `kernel` — тип ядра;
- `losses` — схема весов классов (ошибка неправильной классификации классов);
- `normalize` — использовать ли L2-нормализацию входов;
- `backend` — `numpy` или `numba`;
- `compute_dtype` — `auto`, `float32`, `float64`.

### `AdaptivePNN`

Дополнительно поддерживает:

- `loss` — целевая функция оптимизации;
- `bandwidth_sharing` — способ задания параметров ширины;
- `max_iter` — максимальное число шагов оптимизации;
- `solver` — метод оптимизации;
- `solver_options` — дополнительные параметры оптимизатора.

### `GRNN`

Основные параметры:

- `bandwidth`;
- `kernel`;
- `backend`;
- `compute_dtype`;
- `normalize`.

### `AdaptiveGRNN`

Дополнительно поддерживает:

- `loss` — функция ошибки для регрессии;
- `max_iter`;
- `solver`;
- `solver_options`;
- `normalize`.

`AdaptiveGRNN` оптимизирует ширину ядра по схеме `per_feature`.

## Тестирование

Предпочтительный способ запуска тестов:

```bash
tox
```

Или запуск конкретного файла тестов:

```bash
pytest tests/test_pnn.py
```

Так как тесты импортируют установленный пакет, удобнее всего запускать их в подготовленном окружении после установки зависимостей.

## Производительность

В репозитории есть отдельные материалы для измерения производительности:

- [benchmarks/inference_bemchmarks.py](benchmarks/inference_bemchmarks.py)
- [benchmarks/benchmarking.ipynb](benchmarks/benchmarking.ipynb)

## Примеры

Дополнительный пример использования находится в:

- [examples/basic_usage.ipynb](examples/basic_usage.ipynb)

## Лицензия
Проект распространяется по лицензии MIT. См. файл [LICENCE](LICENCE).
