"""Watches a training run from outside the training loop: checkpoints, loss curves and optional W&B logging."""

import hashlib
import json
import os
import socket
import time

import torch


def online(host: str = "api.wandb.ai", port: int = 443, timeout: float = 3.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def now():
    return time.strftime("%H:%M:%S")


class TrainingRunMonitor:
    """
    Checkpoints, loss-curve log and W&B run of one training run; the training itself is done by the caller.

    Crash / restart: load() returns the last checkpoint and the W&B run id is
    derived from (out_dir, key), so a rerun continues the same run.
    Lost connection: every `check_every` epochs api.wandb.ai is checked; if
    unreachable, a checkpoint is written and training pauses until it is back.
    """

    def __init__(self, out_dir, key, config, use_wandb=True, project="mouldit_benchmark",
                 entity=None, group=None, check_every=10, ckpt_every=25):
        self.key, self.use_wandb = key, use_wandb
        self.check_every, self.ckpt_every = check_every, ckpt_every
        self.ckpt_path  = os.path.join(out_dir, "checkpoints", f"{key}.pt")
        self.curve_path = os.path.join(out_dir, "loss_curves", f"{key}.json")
        for p in (self.ckpt_path, self.curve_path):
            os.makedirs(os.path.dirname(p), exist_ok=True)
        self.run = None
        if use_wandb:
            import wandb
            self.wandb = wandb
            run_id = hashlib.md5(f"{os.path.abspath(out_dir)}|{key}".encode()).hexdigest()[:16]
            while self.run is None:
                self.wait_online()
                try:
                    self.run = wandb.init(project=project, entity=entity, id=run_id, resume="allow",
                                          group=group, name=key, config=config, reinit=True,
                                          dir=out_dir, settings=wandb.Settings(init_timeout=120))
                except Exception as e:
                    print(f"[{now()}] wandb.init failed ({e}); retrying", flush=True)
                    time.sleep(30)
            self.run.define_metric("epoch")
            self.run.define_metric("train/*", step_metric="epoch")

    def wait_online(self):
        if not self.use_wandb or online():
            return
        print(f"[{now()}] {self.key}: W&B unreachable — paused", flush=True)
        while not online():
            time.sleep(60)
        print(f"[{now()}] {self.key}: connection back — resuming", flush=True)

    def load(self):
        if os.path.exists(self.ckpt_path + ".tmp"):
            os.remove(self.ckpt_path + ".tmp")
        if not os.path.exists(self.ckpt_path):
            return None
        print(f"[{now()}] {self.key}: resuming from checkpoint", flush=True)
        return torch.load(self.ckpt_path, map_location="cpu", weights_only=False)

    def save(self, state):
        torch.save(state, self.ckpt_path + ".tmp")
        os.replace(self.ckpt_path + ".tmp", self.ckpt_path)

    def epoch_end(self, epoch, losses, state_fn):
        if self.run:
            self.run.log({"epoch": epoch, **{f"train/{k}": v for k, v in losses.items()}})
        if (epoch + 1) % self.ckpt_every == 0:
            self.save(state_fn())
        if self.use_wandb and (epoch + 1) % self.check_every == 0 and not online():
            self.save(state_fn())
            self.wait_online()

    def finish(self, loss_log, records):
        """Write the loss curves, log *records* (dicts with a "method" key) and drop the checkpoint."""
        json.dump(loss_log, open(self.curve_path, "w"))
        if self.run:
            self.run.summary.update({f"{r['method']}/{k}": v for r in records
                                     for k, v in r.items() if isinstance(v, float)})
            self.wait_online()
            self.run.finish()
        if os.path.exists(self.ckpt_path):
            os.remove(self.ckpt_path)
