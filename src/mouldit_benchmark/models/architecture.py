"""
Builds MouldIt's encoder and decoder from a YAML file in models/configs/.

configs/tcn_attention.yaml is the architecture from the paper and the default.  A YAML file lists
the layers in order, one per line.  To try a different design, copy it under a new name, change the
layers, and pass that name as `architecture` in the benchmark config.

Sizes that depend on the data can be written as small formulas, for example "${32 * seq_len}".
You can use seq_len (time steps per cycle), n_channels (number of sensors), code (latent size)
and any name you define under `derived` at the top of the file.

Available layers: linear, conv1d, tcn (temporal convolution block), attention (temporal
attention) and flatten.  Linear and conv1d layers get a LeakyReLU after them, except the very
last layer.  Add `norm: true` to a layer to normalise its output.
"""

import ast
import operator
import re
from dataclasses import dataclass
from pathlib import Path

import torch.nn as nn
import yaml

from .layers import TemporalAttention, TemporalConvBlock

CONFIG_DIR = Path(__file__).parent / "configs"


def _linear(spec):
    return nn.Linear(spec["in_features"], spec["out_features"])


def _conv1d(spec):
    return nn.Conv1d(spec["in_channels"], spec["out_channels"], spec["kernel_size"],
                     stride=spec.get("stride", 1), padding=spec.get("padding", 0))


def _tcn(spec):
    if spec["out_channels"] == 1:
        raise ValueError("a tcn layer needs out_channels > 1, its LayerNorm is over channels")
    return TemporalConvBlock(spec["in_channels"], spec["out_channels"], spec["kernel_size"],
                             dilation=spec.get("dilation", 1))


def _attention(spec):
    return TemporalAttention(spec["hidden_dim"], seq_len=spec.get("seq_len"))


def _flatten(spec):
    return nn.Flatten()


LAYERS = {"linear": _linear, "conv1d": _conv1d, "tcn": _tcn, "attention": _attention, "flatten": _flatten}
ACTIVATED = {"linear", "conv1d"}
NORMS = {"linear": lambda spec: nn.LayerNorm(spec["out_features"]),
         "conv1d": lambda spec: nn.BatchNorm1d(spec["out_channels"]),
         "tcn": lambda spec: nn.BatchNorm1d(spec["out_channels"])}


@dataclass
class Architecture:
    """Layer specs of the encoder and decoder, with every expression resolved to an integer."""

    encoder: list
    decoder: list

    @classmethod
    def from_yaml(cls, name: str, seq_len: int, n_channels: int, code: int):
        """configs/<name>.yaml, or the YAML file `name` itself."""
        path = Path(name) if name.endswith((".yaml", ".yml")) else CONFIG_DIR / f"{name}.yaml"
        cfg = yaml.safe_load(path.read_text())
        env = {"seq_len": seq_len, "n_channels": n_channels, "code": code}
        for key, value in (cfg.get("derived") or {}).items():
            env[key] = _Expressions.resolve(value, env)
        return cls(_Expressions.resolve(cfg["encoder"], env), _Expressions.resolve(cfg["decoder"], env))

    @property
    def hidden_dim(self) -> int:
        """Width of the encoder output, which the latent heads project from."""
        return self.encoder[-1]["out_features"]

    def build_encoder(self) -> nn.Sequential:
        return self.build(self.encoder)

    def build_decoder(self) -> nn.Sequential:
        return self.build(self.decoder)

    @staticmethod
    def build(specs) -> nn.Sequential:
        layers = []
        for i, spec in enumerate(specs):
            kind = spec["type"]
            if kind not in LAYERS:
                raise ValueError(f"layer {i}: unknown type {kind!r}")
            layers.append(LAYERS[kind](spec))
            if kind in ACTIVATED and i < len(specs) - 1:
                layers.append(nn.LeakyReLU(0.1))
            if spec.get("norm") and kind in NORMS:
                layers.append(NORMS[kind](spec))
        return nn.Sequential(*layers)


class _Expressions:
    """Resolves "${expr}" strings: integer + - * // over named dimensions."""

    PATTERN = re.compile(r"^\$\{(.+)\}$")
    OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv}

    @classmethod
    def resolve(cls, value, env):
        if isinstance(value, list):
            return [cls.resolve(v, env) for v in value]
        if isinstance(value, dict):
            return {k: cls.resolve(v, env) for k, v in value.items()}
        if isinstance(value, str) and (m := cls.PATTERN.match(value)):
            return cls.evaluate(ast.parse(m.group(1), mode="eval").body, env)
        return value

    @classmethod
    def evaluate(cls, node, env):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in cls.OPS:
            return cls.OPS[type(node.op)](cls.evaluate(node.left, env), cls.evaluate(node.right, env))
        raise ValueError(f"unsupported expression: {ast.unparse(node)}")
