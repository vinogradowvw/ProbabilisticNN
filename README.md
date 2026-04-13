# ProbabilisticNN

- `PNN` — классическая Probabilistic Neural Network для классификации.
- `AdaptivePNN` — версия PNN, где ширина ядра подбирается оптимизацией.
- `GRNN` — General Regression Neural Network для задач регрессии.
- `examples/basic_usage.ipynb` — ноутбук с примерами использования и сравнением обычного PNN с AdaptivePNN.
- `tests/` — smoke-тесты, которые проверяют, что основные части проекта не сломались.

## Структура проекта

```text
src/
  base/
    kernels.py        # gaussian kernel и torch/numpy версии
    optim.py          # оптимизация bandwidth для AdaptivePNN
    utils.py          # общие утилиты, например L2-нормализация
    types.py          # типы для callable kernel

  common/
    pattern_layer.py  # общий pattern layer для PNN/GRNN и adaptive-вариант

  pnn/
    pnn.py            # PNN и AdaptivePNN как sklearn-like estimators
    layers.py         # summation layer и output layer для классификации

  grnn/
    grnn.py           # GRNN estimator
    summation_layer.py # слой агрегации для регрессии

examples/
  basic_usage.ipynb   # основной пример запуска

tests/
  test_pnn_smoke.py   # базовые тесты для PNN/AdaptivePNN
  test_grnn_smoke.py  # базовые тесты для GRNN
```

Разделение сделано так, чтобы не смешивать математику, слои и готовые модели:

- `base` содержит оптимизацию параметров и ядровые функции, которые могут использовать разные модели.
- `common` содержит общий pattern layer, потому что PNN и GRNN оба работают через одинаковые pattern layer.
- `pnn` содержит только классификационную часть.
- `grnn` содержит только регрессионную часть.
- `examples` нужен для запуска и визуальной проверки.

## Коротко про PNN

PNN хранит обучающие объекты как паттерны. Для нового объекта считается похожесть до всех обучающих объектов через Gaussian kernel. Потом ответы суммируются по классам, и выбирается класс с максимальной апостериорной оценкой.

Главный параметр обычного PNN — `bandwidth`. Он отвечает за степень сглаживания:

- маленький `bandwidth` делает модель более локальной;
- большой `bandwidth` делает модель более гладкой.

## Коротко про AdaptivePNN

В AdaptivePNN ширина ядра не задается одной константой, а подбирается оптимизацией.

Основной вариант в проекте:

```python
AdaptivePNN(bandwidth_sharing="per_feature")
```

Это значит, что у каждого признака есть своя ширина. Такой вариант ближе к подходу Specht: разные признаки могут иметь разную полезность и разную форму распределения.

Также есть варианты с подходами по разделению ширины на классы, и на классы и признаки одновременно.