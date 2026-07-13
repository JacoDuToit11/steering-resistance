"""Activation-steering injection hook (spec §3.5).

Adds alpha * vec to the residual stream output of a decoder layer, at all
token positions, during any forward pass (training or eval). The scaling by
mean residual norm is baked into `vec` at construction time (see vectors.py),
so `alpha` here is the dimensionless strength.
"""

from contextlib import contextmanager

import torch


def make_hook(vec: torch.Tensor, alpha: float):
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden + alpha * vec.to(hidden.dtype).to(hidden.device)
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    return hook


@contextmanager
def steering(layer_module: torch.nn.Module, vec: torch.Tensor, alpha: float):
    """Context manager: inject vec at `layer_module` output while active.

    `layer_module` must be the decoder layer itself (from get_decoder_layers),
    not a guessed attribute path.
    """
    handle = layer_module.register_forward_hook(make_hook(vec, alpha))
    try:
        yield
    finally:
        handle.remove()


def make_batch_hook(vecs: torch.Tensor, alphas: torch.Tensor):
    """Per-row injection: row i of the batch gets alphas[i] * vecs[i].

    vecs: [B, d_model], alphas: [B]. Rows with alpha 0 (e.g. clean rows mixed
    into a steered batch) are untouched. Broadcasts over the sequence dim, so
    it works for both prefill [B, seq, d] and decode [B, 1, d] passes.
    """

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        add = (alphas[:, None, None] * vecs[:, None, :]).to(hidden.dtype).to(hidden.device)
        hidden = hidden + add
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    return hook


@contextmanager
def steering_batch(layer_module: torch.nn.Module, vecs: torch.Tensor, alphas: torch.Tensor):
    """Context manager: per-row batched injection while active (see make_batch_hook)."""
    handle = layer_module.register_forward_hook(make_batch_hook(vecs, alphas))
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def capture_residual(layer_module: torch.nn.Module, store: list):
    """Capture the residual-stream output of a layer into `store` (appended)."""

    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        store.append(hidden.detach())

    handle = layer_module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
