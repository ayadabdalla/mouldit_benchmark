"""Training and inference of MouldIt, with the same interface as the baselines (fit, score, predict, reconstruct)."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..datasets import CHANNELS, DATA_DIR, ReferencePreprocessing, sigma_ok
from .mouldit import MouldIt
from .novelty import ClassGaussians

LOSS_KEYS = ["total", "mse", "kld", "cls", "train_acc"]


class MouldItRunner:
    """
    Loss: recon_weight (MSE + beta(e) KLD) + cls_weight BCE, with beta(e) = beta min(1, e / 100)
    and beta from kld_beta.
    Adam with cosine annealing to 1e-5, gradient norm clipped to 1.

    prediction = class with the highest head output
    novelty    = max(L, R / eta_R) on mu (see novelty_terms)
    """

    def __init__(self, beta: float, cls_weight: float = 0.1, epochs: int = 1000, weight_decay: float = 0.0,
                 lr: float = 1e-3, batch_size: int = 32, recon_weight: float = 1.0, code: int = 32,
                 quantile: float = 95, random_state: int = 42, architecture: str = "tcn_attention",
                 device: str | None = None):
        self.cls_weight, self.epochs, self.beta = cls_weight, epochs, beta
        self.weight_decay, self.lr, self.batch_size = weight_decay, lr, batch_size
        self.recon_weight, self.code, self.quantile = recon_weight, code, quantile
        self.random_state, self.architecture = random_state, architecture
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def kld_beta(channels=CHANNELS, n: int = 300, seed: int = 42, data_dir=DATA_DIR) -> float:
        """
        beta = 2 sigma_OK^2 / (C T) (paper: MouldIt Model), with sigma_OK the RMS deviation of the first n
        synthetic OK cycles at `seed` from the reference and C T the values per cycle.
        configs/benchmark.yaml holds it for each setting, rounded to two significant digits.
        """
        values_per_cycle = ReferencePreprocessing(data_dir).load(channels)[1].size
        return 2 * sigma_ok(channels, n, seed, data_dir) ** 2 / values_per_cycle

    def build(self, X, n_classes):
        self.model = MouldIt.from_config(self.architecture, X.shape[2], X.shape[1], self.code,
                                                n_classes).to(self.device)
        return self.model

    def fit(self, X, y, X_val, y_val, monitor=None):
        """
        Train on (X, y), then calibrate the novelty score on (X_val, y_val).
        monitor (optional, see training_run_monitor.TrainingRunMonitor): load() -> checkpoint or None, and
        epoch_end(epoch, losses, state_fn) for logging and checkpointing.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        n_classes = int(y.max()) + 1
        model = self.build(X, n_classes)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-5)
        onehot = F.one_hot(torch.from_numpy(y), n_classes).float()
        loader = DataLoader(TensorDataset(torch.from_numpy(X), onehot), batch_size=self.batch_size, shuffle=True)
        self.loss_log = {k: [] for k in LOSS_KEYS}
        start = 0

        ckpt = monitor.load() if monitor else None
        if ckpt is not None:
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["opt"])
            sched.load_state_dict(ckpt["sched"])
            self.loss_log = ckpt["loss_log"]
            torch.set_rng_state(ckpt["rng_torch"])
            np.random.set_state(ckpt["rng_np"])
            if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(ckpt["rng_cuda"])
            start = ckpt["epoch"] + 1

        def state(epoch):
            return {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": epoch, "loss_log": self.loss_log, "rng_torch": torch.get_rng_state(),
                    "rng_np": np.random.get_state(),
                    "rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None}

        for epoch in range(start, self.epochs):
            model.train()
            sums, n = dict.fromkeys(LOSS_KEYS, 0.0), 0
            beta = min(self.beta, self.beta * epoch / 100)
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                recon, prob, mu, logvar = model(xb)
                mse = F.mse_loss(recon, xb)
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
                bce = F.binary_cross_entropy(prob, yb)
                loss = self.recon_weight * (mse + beta * kld) + self.cls_weight * bce
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                acc = (prob.argmax(1) == yb.argmax(1)).float().mean()
                for k, v in zip(LOSS_KEYS, (loss, mse, kld, bce, acc)):
                    sums[k] += v.item()
                n += 1
            sched.step()
            for k in sums:
                self.loss_log[k].append(sums[k] / n)
            if monitor:
                monitor.epoch_end(epoch, {k: v[-1] for k, v in self.loss_log.items()}
                                  | {"beta": beta, "lr": sched.get_last_lr()[0]}, lambda: state(epoch))
        model.eval()
        return self.calibrate(X, y, X_val, y_val)

    def load(self, weights, X, y, X_val, y_val):
        """Load trained weights, then calibrate the novelty score as fit() does."""
        self.build(X, int(y.max()) + 1).load_state_dict(torch.load(weights, map_location=self.device))
        self.model.eval()
        return self.calibrate(X, y, X_val, y_val)

    def calibrate(self, X, y, X_val, y_val):
        """Per-class Gaussians on the training mu, eta_q and eta_R on the validation cycles."""
        mu_tr, _, _ = self._outputs(X)
        mu_va, rec_va, _ = self._outputs(X_val)
        self.gaussians = ClassGaussians(self.quantile).fit(mu_tr, y, mu_va, y_val)
        self.eta_r = np.percentile(squared_error(X_val, rec_va), self.quantile)
        return self

    @torch.no_grad()
    def _outputs(self, X, batch: int = 256):
        """mu, the reconstruction from mu, and the class probabilities."""
        mus, recs, probs = [], [], []
        for s in range(0, len(X), batch):
            xb = torch.from_numpy(X[s:s + batch]).to(self.device)
            mu, _ = self.model.encode(xb)
            recs.append(self.model.decode(mu, xb.shape).cpu().numpy())
            probs.append(self.model.classify(mu).cpu().numpy())
            mus.append(mu.cpu().numpy())
        return np.concatenate(mus), np.concatenate(recs), np.concatenate(probs)

    def novelty_terms(self, X):
        """(N, Q) normalised class distances M_q / eta_q of mu, and the reconstruction term R / eta_R."""
        mu, rec, _ = self._outputs(X)
        return self.gaussians.distances(mu), squared_error(X, rec) / self.eta_r

    def score(self, X):
        mu, rec, prob = self._outputs(X)
        latent = self.gaussians.score(mu)[0]
        return prob.argmax(1), np.maximum(latent, squared_error(X, rec) / self.eta_r)

    def predict(self, X):
        return self._outputs(X)[2].argmax(1)

    def transform(self, X):
        return self._outputs(X)[0]

    def reconstruct(self, X):
        return self._outputs(X)[1]


def squared_error(X, R):
    """Summed squared error per cycle."""
    return ((X - R) ** 2).reshape(len(X), -1).sum(1)
