"""A small neural network written from scratch with numpy only.

No machine-learning libraries: the forward pass, backpropagation and the Adam
optimizer are all written out below so you can read and learn from them.

    MLP       - fully connected network (tanh hidden layers, linear output)
    Regressor - wraps an MLP with input/output standardization, training loop,
                early stopping and save/load to a .npz file
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


class MLP:
    def __init__(self, sizes: list[int], seed: int = 0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            # Xavier/Glorot initialization keeps tanh activations in a healthy range.
            limit = np.sqrt(6.0 / (n_in + n_out))
            self.W.append(rng.uniform(-limit, limit, size=(n_in, n_out)))
            self.b.append(np.zeros(n_out))

    # ----------------------------------------------------------------- forward
    def forward(self, X: np.ndarray, keep: bool = False):
        """Return the output; with keep=True also the activations needed for backprop."""
        acts = [X]
        h = X
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = z if i == last else np.tanh(z)
            acts.append(h)
        return (h, acts) if keep else h

    # ---------------------------------------------------------------- backward
    def backward(self, acts: list[np.ndarray], grad_out: np.ndarray):
        """Backpropagation: gradients of the loss w.r.t. every weight and bias."""
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        delta = grad_out                      # dLoss/dz for the (linear) output layer
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i > 0:
                # Through the weights, then through tanh:  d tanh(z)/dz = 1 - tanh(z)^2
                delta = (delta @ self.W[i].T) * (1.0 - acts[i] ** 2)
        return gW, gb

    def params(self):
        return self.W + self.b


class Adam:
    """Adam optimizer (Kingma & Ba, 2014)."""

    def __init__(self, params: list[np.ndarray], lr: float = 1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads: list[np.ndarray]):
        self.t += 1
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class Regressor:
    """Standardizes data, trains an MLP with mini-batch Adam on mean-squared error."""

    def __init__(self, n_in: int, n_out: int, hidden=(64, 64), seed: int = 0):
        self.net = MLP([n_in, *hidden, n_out], seed=seed)
        self.x_mu = np.zeros(n_in)
        self.x_sd = np.ones(n_in)
        self.y_mu = np.zeros(n_out)
        self.y_sd = np.ones(n_out)
        self.meta: dict = {}
        self.history: list[tuple[int, float, float]] = []

    def fit(self, X, Y, epochs=3000, lr=3e-3, batch_size=128, val_frac=0.15, l2=1e-5,
            patience=300, seed=0, verbose=True, progress=None):
        rng = np.random.default_rng(seed)
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        order = rng.permutation(len(X))
        n_val = max(1, int(len(X) * val_frac))
        val, tr = order[:n_val], order[n_val:]
        self.x_mu, self.x_sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        self.y_mu, self.y_sd = Y[tr].mean(0), Y[tr].std(0) + 1e-9
        Xs, Ys = (X - self.x_mu) / self.x_sd, (Y - self.y_mu) / self.y_sd

        opt = Adam(self.net.params(), lr=lr)
        best, best_params, since_best = np.inf, None, 0
        for epoch in range(1, epochs + 1):
            perm = rng.permutation(tr)
            for s in range(0, len(perm), batch_size):
                bi = perm[s:s + batch_size]
                pred, acts = self.net.forward(Xs[bi], keep=True)
                grad = 2.0 * (pred - Ys[bi]) / len(bi)          # d(MSE)/d(pred)
                gW, gb = self.net.backward(acts, grad)
                gW = [g + l2 * W for g, W in zip(gW, self.net.W)]  # weight decay
                opt.step(gW + gb)
            if epoch % 10 == 0 or epoch == 1:
                tr_loss = float(np.mean((self.net.forward(Xs[tr]) - Ys[tr]) ** 2))
                va_loss = float(np.mean((self.net.forward(Xs[val]) - Ys[val]) ** 2))
                self.history.append((epoch, tr_loss, va_loss))
                if va_loss < best - 1e-7:
                    best, since_best = va_loss, 0
                    best_params = [p.copy() for p in self.net.params()]
                else:
                    since_best += 10
                if verbose and epoch % 200 == 0:
                    print(f"  epoch {epoch:5d}  train MSE {tr_loss:.5f}  val MSE {va_loss:.5f}")
                if progress:
                    progress(epoch, epochs, va_loss)
                if since_best >= patience:
                    if verbose:
                        print(f"  early stop at epoch {epoch}")
                    break
        if best_params is not None:
            for p, bp in zip(self.net.params(), best_params):
                p[...] = bp
        # Validation error in the original units (for reporting).
        val_pred = self.predict(X[val])
        self.meta["val_mae"] = np.abs(val_pred - Y[val]).mean(axis=0).tolist()
        return self

    def predict(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        return self.net.forward((X - self.x_mu) / self.x_sd) * self.y_sd + self.y_mu

    # ------------------------------------------------------------------- I/O
    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"W{i}": W for i, W in enumerate(self.net.W)}
        arrays.update({f"b{i}": b for i, b in enumerate(self.net.b)})
        np.savez(path, x_mu=self.x_mu, x_sd=self.x_sd, y_mu=self.y_mu, y_sd=self.y_sd,
                 sizes=np.array(self.net.sizes), meta=np.array(json.dumps(self.meta)), **arrays)

    @classmethod
    def load(cls, path: Path) -> "Regressor":
        d = np.load(path, allow_pickle=False)
        sizes = d["sizes"].tolist()
        reg = cls(sizes[0], sizes[-1], hidden=tuple(sizes[1:-1]))
        reg.net.W = [d[f"W{i}"] for i in range(len(sizes) - 1)]
        reg.net.b = [d[f"b{i}"] for i in range(len(sizes) - 1)]
        reg.x_mu, reg.x_sd, reg.y_mu, reg.y_sd = d["x_mu"], d["x_sd"], d["y_mu"], d["y_sd"]
        reg.meta = json.loads(str(d["meta"]))
        return reg


def gradient_check(seed: int = 0) -> float:
    """Compare backprop gradients with finite differences (should be ~1e-7 or smaller)."""
    rng = np.random.default_rng(seed)
    net = MLP([3, 5, 2], seed=seed)
    X, Y = rng.normal(size=(4, 3)), rng.normal(size=(4, 2))

    def loss():
        return float(np.mean((net.forward(X) - Y) ** 2))

    pred, acts = net.forward(X, keep=True)
    gW, gb = net.backward(acts, 2 * (pred - Y) / Y.size)
    worst = 0.0
    eps = 1e-6
    for p, g in zip(net.params(), gW + gb):
        it = np.nditer(p, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = p[i]
            p[i] = old + eps
            lp = loss()
            p[i] = old - eps
            lm = loss()
            p[i] = old
            worst = max(worst, abs((lp - lm) / (2 * eps) - g[i]))
    return worst
