"""
Erlanger Träger benchmark: MouldIt and the baselines on every modality, dataset size, seed and fold.

    python scripts/run_benchmark.py                             # the full sweep of configs/benchmark.yaml
    python scripts/run_benchmark.py totals=[500] seeds=[42]     # override any config value
    python scripts/run_benchmark.py --config other.yaml

Writes to results/erlanger_traeger_benchmark/ (config `out`):
    config.yaml        the settings; a rerun with different method settings is refused
    results.json       one record per fold and method
    outputs/<fold>.npz per-cycle predictions and novelty scores of every method; the sorting errors,
                       including those of RF + PCA flag, are computed from these by report_benchmark.py
    models/<fold>.pt   trained MouldIt weights
A rerun resumes: finished records are skipped and an interrupted MouldIt training continues from its
checkpoint.  `touch <out>/STOP` stops cleanly before the next fold.
"""

import argparse
import json
import os
import subprocess

import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from report_benchmark import report
from mouldit_benchmark.datasets import MODALITIES, SyntheticDataset, n_classes
from mouldit_benchmark.evaluation import MethodEvaluator
from mouldit_benchmark.models import KNN, PCA, LogisticRegression, MouldItRunner, RandomForest, Reference
from mouldit_benchmark.paths import CONFIG, RESULTS_DIR
from mouldit_benchmark.training_run_monitor import TrainingRunMonitor, now

METHOD_KEYS = ("quantile", "code_dim", "knn", "mould_it", "fidelity")   # settings that change results


def load_config():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("overrides", nargs="*", help="key=value, e.g. totals=[500] mould_it.epochs=5")
    args = ap.parse_args()
    cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(cfg, True)
    try:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    except ConfigKeyError as e:
        ap.error(f"unknown config key {e.full_key!r}")
    cfg.out = cfg.out or str(RESULTS_DIR)
    for m in cfg.modalities:
        if m not in MODALITIES:
            ap.error(f"unknown modality {m!r}, choose from {list(MODALITIES)}")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        ap.error("device=cuda but CUDA is not available; pass device=cpu to train on the CPU")
    return cfg


def check_resume(cfg, path):
    """Refuse to add to results produced with different method settings."""
    if os.path.exists(path):
        old = OmegaConf.load(path)
        changed = [k for k in METHOD_KEYS if old.get(k) != cfg.get(k)]
        if changed:
            raise SystemExit(f"{path} was written with different {', '.join(changed)}; "
                             "use another out= or the same settings")
    OmegaConf.save(cfg, path)


def baselines(cfg, seed):
    return {
        "Reference": Reference(),
        "kNN-raw": KNN(n_neighbors=cfg.knn.n_neighbors),   # the paper's kNN; key kept to match stored results
        "LogReg": LogisticRegression(random_state=seed),
        "RF": RandomForest(random_state=seed),
        "PCA": PCA(cfg.code_dim, cfg.quantile, random_state=seed),
    }


def save_json(records, path):
    json.dump(records, open(path + ".tmp", "w"), indent=1)
    os.replace(path + ".tmp", path)


def save_outputs(path, y_te, outputs):
    """Add <method>__<array> for the given methods to the fold's npz."""
    arrays = dict(np.load(path)) if os.path.exists(path) else {}
    arrays["y_test"] = y_te
    arrays |= {f"{m}__{k}": v for m, o in outputs.items() for k, v in o.items()}
    np.savez_compressed(path + ".tmp.npz", **arrays)
    os.replace(path + ".tmp.npz", path)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).strip()
    except Exception:
        return "unknown"


class Benchmark:
    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(os.path.join(cfg.out, "outputs"), exist_ok=True)
        os.makedirs(os.path.join(cfg.out, "models"), exist_ok=True)
        check_resume(cfg, os.path.join(cfg.out, "config.yaml"))
        self.results_path = os.path.join(cfg.out, "results.json")
        self.records = json.load(open(self.results_path)) if os.path.exists(self.results_path) else []
        self.done = {self.key(r) for r in self.records}
        if self.done:
            print(f"[{now()}] resuming: {len(self.done)} records in {self.results_path}", flush=True)
        self.stop_path = os.path.join(cfg.out, "STOP")
        if os.path.exists(self.stop_path):
            os.remove(self.stop_path)

    @staticmethod
    def key(r):
        return r["modality"], r["total"], r["seed"], r["holdout"], r["method"]

    def stop_requested(self):
        if not os.path.exists(self.stop_path):
            return False
        os.remove(self.stop_path)
        print(f"[{now()}] stopped; rerun the same command to resume", flush=True)
        return True

    def add(self, records):
        self.records += records
        save_json(self.records, self.results_path)

    def run(self):
        cfg = self.cfg
        for modality in cfg.modalities:
            channels = MODALITIES[modality]
            k = n_classes(channels)
            for total in cfg.totals:
                for seed in cfg.seeds:
                    ds = SyntheticDataset(channels, total // k, seed)
                    for holdout in [None, *(cfg.holdout or ds.classes[1:])]:
                        if self.stop_requested():
                            return False
                        fold = {"modality": modality, "total": total, "n_cycles": (total // k) * k,
                                "seed": seed, "holdout": holdout}
                        evaluator = MethodEvaluator(ds, modality, total, seed, holdout, cfg.quantile,
                                                    cfg.fidelity.dtw_band)
                        new = self.run_baselines(fold, evaluator)
                        if self.key(fold | {"method": "mould_it"}) not in self.done:
                            if self.stop_requested():
                                return False
                            new.append(self.run_mouldit(fold, evaluator, new))
                        if new:
                            self.log(fold, new)
        return True

    def run_baselines(self, fold, evaluator):
        """RF + PCA flag needs no run of its own: ResultsAnalyzer combines RF's and PCA's saved outputs."""
        (X_tr, y_tr), (X_va, y_va), (_, y_te) = evaluator.train, evaluator.validation, evaluator.test
        new, outputs = [], {}
        for name, method in baselines(self.cfg, fold["seed"]).items():
            if self.key(fold | {"method": name}) in self.done:
                continue
            method.fit(X_tr, y_tr, X_va, y_va)
            record, outputs[name] = evaluator.evaluate(method)
            new.append(fold | {"method": name} | record)
        if new:
            self.add(new)
        if outputs:
            save_outputs(self.outputs_path(evaluator), y_te, outputs)
        return new

    def run_mouldit(self, fold, evaluator, baseline_records):
        cfg = self.cfg
        (X_tr, y_tr), (X_va, y_va), (_, y_te) = evaluator.train, evaluator.validation, evaluator.test
        name = evaluator.name
        params = {**OmegaConf.to_container(cfg.mould_it), "beta": cfg.mould_it.beta[fold["modality"]]}
        config = {**fold, **params, "code_dim": cfg.code_dim, "quantile": cfg.quantile, "git_commit": git_commit()}
        monitor = TrainingRunMonitor(cfg.out, name, config, use_wandb=cfg.wandb.enabled, project=cfg.wandb.project,
                                     entity=cfg.wandb.entity, group=os.path.basename(os.path.normpath(cfg.out)))
        model = MouldItRunner(**params, code=cfg.code_dim, quantile=cfg.quantile, random_state=fold["seed"],
                              device=cfg.device)
        model.fit(X_tr, y_tr, X_va, y_va, monitor=monitor)
        record, outputs = evaluator.evaluate(model)
        record = fold | {"method": "mould_it"} | record
        self.add([record])
        save_outputs(self.outputs_path(evaluator), y_te, {"mould_it": outputs})
        weights = os.path.join(cfg.out, "models", f"{name}.pt")
        torch.save(model.model.state_dict(), weights + ".tmp")
        os.replace(weights + ".tmp", weights)
        monitor.finish(model.loss_log, [*baseline_records, record])
        return record

    def outputs_path(self, evaluator):
        return os.path.join(self.cfg.out, "outputs", evaluator.name + ".npz")

    @staticmethod
    def log(fold, new):
        metric = "accuracy" if fold["holdout"] is None else "auroc"
        task = "classification" if fold["holdout"] is None else f"holdout {fold['holdout']}"
        print(f"[{now()}] {fold['modality']} n={fold['total']} seed={fold['seed']} {task}  {metric}: "
              + "  ".join(f"{r['method']}={r[metric]:.2f}" for r in new if metric in r), flush=True)


if __name__ == "__main__":
    cfg = load_config()
    if Benchmark(cfg).run():
        report(cfg.out)
