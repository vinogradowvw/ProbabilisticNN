from pathlib import Path
import sys
import numpy as np
import time

from probabilisticnn.pnn import PNN, AdaptivePNN
from probabilisticnn.grnn import GRNN, AdaptiveGRNN


np.random.seed(42)

def get_dataset(
        n_train=1000,
        n_test=1000,
        n_features=20,
        task="classification",
        dtype=np.dtype("float64")
):
    rng = np.random.default_rng(42)
    X_train = rng.random((n_train, n_features), dtype=dtype)
    X_test = rng.random((n_test, n_features), dtype=dtype)
    if task == "classification":
        y = rng.integers(0, 2, size=n_train)
    elif task == "regression":
        y = (
            np.sin(2.0 * X_train[:, 0])
            + 0.25 * X_train[:, min(1, n_features - 1)]
            + 0.05 * rng.normal(size=n_train)
        ).astype(dtype, copy=False)
    else:
        raise ValueError(f"Unknown task: {task}")
    return X_train, X_test, y

def resolve_model(model_name: str = "pnn", bandwidth_sharing: str | None = None, backend: str = "numpy"):
    if model_name == "pnn":
        return PNN(backend=backend)
    elif model_name == "grnn":
        return GRNN(backend=backend)
    elif model_name == "adaptive_grnn":
        return AdaptiveGRNN(backend=backend)
    elif model_name == "adaptive_pnn":
        return AdaptivePNN(bandwidth_sharing=bandwidth_sharing, backend=backend)
    else:
        raise ValueError(f"Unknown model: {model_name}")

def measure_inference(model, X):
    # первый прогон для numba jit компиляции
    y_pred = model.predict(X)

    start = time.perf_counter()
    y_pred = model.predict(X)
    end = time.perf_counter()
    return end - start


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", choices=["pnn", "grnn", "adaptive_grnn", "adaptive_pnn"], default="pnn")
    parser.add_argument("--bandwidth-sharing", choices=["per_feature", "per_class", "per_class_per_feature"], default=None)
    parser.add_argument("--backend", choices=["numpy", "numba"], default="numpy")
    parser.add_argument("--repeats", type=int, default=30)
    return parser.parse_args()

def main():
    args = parse_args()
    model = resolve_model(args.model, args.bandwidth_sharing, args.backend)
    task = "classification" if "pnn" in args.model else "regression"
    X_train, X_test, y = get_dataset(args.n_train, args.n_test, args.n_features, task=task)
    model.fit(X_train, y)


    secs = []

    for _ in range(args.repeats):
        seconds = measure_inference(model, X_test)
        secs.append(seconds)

    print(f"Average inference time: {np.mean(secs):.4f} secs")

if __name__ == "__main__":
    main()