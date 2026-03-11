from typing import Type

import abc
from collections import defaultdict, OrderedDict
from functools import wraps
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel, GenerationMixin


class Tracker(abc.ABC):
  """Base class for tracking model behavior."""

  def __init__(
      self,
      model: Type[GenerationMixin | nn.Module],
      clear_on_generate: bool = True,
  ):
    self.model = model
    self.clear_on_generate = clear_on_generate

    self.original_generate = self.model.generate
    if self.clear_on_generate:
      self._patch_generate()

    self.hooks: list[torch.utils.hooks.RemovableHandle] = []
    self.aggregate_hook = model.register_forward_hook(self._get_aggregate_hook())  # yapf: disable

  def _patch_generate(self):
    """Patch the model's generate method to allow auto-clearing."""

    @wraps(self.model.generate)
    def new_generate(*args, **kwargs):
      self.clear()
      return self.original_generate(*args, **kwargs)

    self.model.generate = new_generate

  @abc.abstractmethod
  def _get_aggregate_hook(self) -> callable:
    """Return a hook function that aggregates data from the model."""
    pass

  @abc.abstractmethod
  def collect(self):
    pass

  @abc.abstractmethod
  def clear(self):
    """Clear the collected data."""
    pass

  def detach(self):
    """Detach all hooks from the model. Reset original generate function."""
    for handle in self.hooks:
      handle.remove()

    # Reset the model's generate function
    self.model.generate = self.original_generate


class LensTracker(Tracker):
  """Tracks logit lens results for each layer."""

  def __init__(
      self,
      model: Type[GenerationMixin | nn.Module],
      probs_hooks: OrderedDict[
          str,
          tuple[nn.Module, torch.utils.hooks.RemovableHandle],
      ],
      clear_on_generate: bool = True,
  ):
    super().__init__(model, clear_on_generate)

    self.probs_hooks = probs_hooks

    for _, (_, handle) in probs_hooks.items():
      self.hooks.append(handle)

    self.probs: dict[list[torch.Tensor]] = defaultdict(list)

  def _get_aggregate_hook(self):

    @torch.no_grad()
    def aggregate_hook(*_):
      for name, (layer, _) in self.probs_hooks.items():
        self.probs[name].append(layer._probs)

    return aggregate_hook

  def collect(self) -> dict[str, torch.Tensor]:
    logit_lenses = dict()
    for name, probs_list in self.probs.items():
      if probs_list:
        logit_lenses[name] = torch.cat(probs_list, dim=1)

    return logit_lenses

  def clear(self):
    self.probs.clear()


class DeepThinkingTokensTracker(Tracker):
  """Tracks deep thinking tokens for each layer, for each token."""

  def __init__(
      self,
      model: Type[GenerationMixin | nn.Module],
      logits_hooks: OrderedDict[
          str,
          tuple[nn.Module, torch.utils.hooks.RemovableHandle],
      ],
      last_layer_name: str,
      clear_on_generate: bool = True,
  ):
    super().__init__(model, clear_on_generate)

    self.logits_hooks = logits_hooks
    self.last_layer_name = last_layer_name

    for _, (_, handle) in logits_hooks.items():
      self.hooks.append(handle)

    self.divergences: dict[list[torch.Tensor]] = defaultdict(list)

  def _get_aggregate_hook(self):

    @torch.no_grad()
    def aggregate_hook(*_):
      final_layer, _ = self.logits_hooks[self.last_layer_name]
      final_logits = final_layer._logits
      final_log_probs = torch.log_softmax(final_logits, dim=-1)
      final_probs = torch.softmax(final_logits, dim=-1)

      for name, (layer, _) in self.logits_hooks.items():
        if name == self.last_layer_name:
          continue

        log_probs = torch.log_softmax(layer._logits, dim=-1)
        probs = torch.softmax(layer._logits, dim=-1)

        # M = log((p1 + p2) / 2)
        M = torch.logsumexp(
            torch.stack([final_log_probs, log_probs], dim=0),
            dim=0,
        ) - torch.log(torch.tensor(2.0, device=probs.device))

        self.divergences[name].append(
            ((F.kl_div(M, probs, reduction='none') +
              F.kl_div(M, final_probs, reduction='none')) / 2).sum(axis=-1))

    return aggregate_hook

  def collect(self) -> dict[str, torch.Tensor]:
    """Calculates J-S Divergence between each layer's output and the final layer's output, for each token."""
    divergence = dict()

    for name, divergences_list in self.divergences.items():
      if divergences_list:
        divergence[name] = torch.cat(divergences_list, dim=1)

    return divergence

  def clear(self):
    self.divergences.clear()


def add_logit_lens_hooks(
    model: PreTrainedModel,
    clear_on_generate: bool = True,
    module_names: list[str] | None = None,
) -> LensTracker:
  """Registers forward hooks on transformer layers and applies logit lens.

  Args:
      model (PreTrainedModel): The model to register hooks on.
      clear_on_generate (bool): Whether to clear collected hidden states on each
          generate call. Defaults to True.
      module_names (list[str] | None): List of module names to register hooks on.
          If None, hooks will be registered on all .layers of the model.
          Defaults to None.

  Returns:
      Tracker: An object to track gating decisions.
  """

  @torch.no_grad()
  def hook(
      module: nn.Module,
      _: torch.Tensor,
      output: torch.Tensor,
  ):
    # Module should be a DecoderLayer
    module._probs = torch.softmax(
        model.lm_head(output),
        dim=-1,
    ).detach().cpu()

  hooks: OrderedDict[
      str,
      tuple[nn.Module, torch.utils.hooks.RemovableHandle],
  ] = OrderedDict()

  for name, layer in model.named_modules():
    if module_names and name in module_names:
      hooks[name] = (layer, layer.register_forward_hook(hook))
    elif re.match(r'.*\.layers\.\d+$', name):
      hooks[name] = (layer, layer.register_forward_hook(hook))

  return LensTracker(
      model,
      hooks,
      clear_on_generate=clear_on_generate,
  )


def add_deep_thinking_tokens_hooks(
    model: PreTrainedModel,
    clear_on_generate: bool = True,
    module_names: list[str] | None = None,
) -> DeepThinkingTokensTracker:
  """Registers forward hooks on transformer layers and applies logit lens.

  Args:
      model (PreTrainedModel): The model to register hooks on.
      clear_on_generate (bool): Whether to clear collected hidden states on each
          generate call. Defaults to True.
      module_names (list[str] | None): List of module names to register hooks on.
          If None, hooks will be registered on all .layers of the model.
          Defaults to None.

  Returns:
      DeepThinkingTokensTracker: An object to track gating decisions.
  """

  @torch.no_grad()
  def hook(
      module: nn.Module,
      _: torch.Tensor,
      output: torch.Tensor,
  ):
    # module should be a DecoderLayer
    module._logits = model.lm_head(output).detach().cpu()

  hooks: OrderedDict[
      str,
      tuple[nn.Module, torch.utils.hooks.RemovableHandle],
  ] = OrderedDict()

  last_layer_name = module_names[-1] if module_names else None

  for name, layer in model.named_modules():
    if module_names and name in module_names:
      hooks[name] = (layer, layer.register_forward_hook(hook))
    elif re.match(r'.*\.layers\.\d+$', name):
      hooks[name] = (layer, layer.register_forward_hook(hook))
      last_layer_name = name

  return DeepThinkingTokensTracker(
      model,
      hooks,
      last_layer_name=last_layer_name,
      clear_on_generate=clear_on_generate,
  )


__all__ = [
    'Tracker',
    'LensTracker',
    'DeepThinkingTokensTracker',
    'add_logit_lens_hooks',
    'add_deep_thinking_tokens_hooks',
]
