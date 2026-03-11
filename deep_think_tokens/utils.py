from typing import Any

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch

logger = logging.getLogger(__name__)


def div_to_matrix(divergences: dict[Any, torch.Tensor]) -> np.ndarray:
  """Convert a dictionary of torch.Tensors to a numpy array

  Args:
      divergences (dict[Any, torch.Tensor]): A dictionary mapping layer names to
        divergence tensors.

  Returns:
      np.ndarray: divergences in matrix form
  """
  div_mat = torch.cat(list(divergences.values()), dim=0).float().numpy()
  if np.isnan(div_mat).any():
    logger.warning("divergences contains NaN values")
    # remove nans
    div_mat = np.nan_to_num(div_mat, nan=0.0, posinf=0.0, neginf=0.0)

  return div_mat


def deep_thinking_ratio(
    divergences: dict[Any, torch.Tensor] | np.ndarray,
    g: float = 0.5,
    p: float = 0.9,
) -> float:
  """Calculate the deep thinking ratio from a dictionary of divergences.

  The deep thinking ratio is defined as the proportion of tokens that have a
  divergence value below 0.5 in the final layer.

  Args:
      divergences (dict[Any, torch.Tensor] | np.ndarray): A dictionary mapping
        layer names to divergence tensors or a numpy array of divergence values.
      g (float): The divergence threshold for considering a token as "deep".
        Defaults to 0.5.
      p (float): The depth fraction threshold for defining a deep-thinking
        regime.

  Returns:
      float: The deep thinking ratio, a value between 0 and 1.
  """
  if isinstance(divergences, dict):
    div_mat = div_to_matrix(divergences)
  else:
    div_mat = divergences

  num_layers, num_tokens = div_mat.shape
  depth_threshold = int(num_layers * p)

  div_mat = np.minimum.accumulate(div_mat)

  mask = div_mat <= g
  deep_idx = np.where(mask.any(axis=0), mask.argmax(axis=0), num_layers)

  deep_thinking_tokens = (deep_idx >= depth_threshold).sum()
  ratio = deep_thinking_tokens / num_tokens

  return ratio.item()


def plot_divergences(
    divergences: dict[str, torch.Tensor] | np.ndarray,
    tokens: list[str] | None,
    fig: plt.Figure | None = None,
    cmap='viridis',
    title: str = 'Divergences',
    accumulate: bool = True,
) -> plt.Figure:
  """Plot divergence values as a heatmap across layers and tokens.

  Args:
    divergences (dict[str, torch.Tensor] | np.ndarray): A dictionary mapping
      layer names to divergence tensors or a numpy array of divergence values.
    tokens (list[str] | None): Optional list of token strings for x-axis labels.
    fig (plt.Figure | None): Optional matplotlib figure to plot on. If None, a
      new figure is created. Defaults to None.
    cmap (str): Colormap for the heatmap. Defaults to 'viridis'.
    title (str): Title for the plot. Defaults to 'Divergences'.
    accumulate (bool): If True, applies cumulative minimum across layers.
      Defaults to True.

  Returns:
    plt.Figure: The matplotlib figure containing the heatmap plot.
  """
  if isinstance(divergences, dict):
    div_mat = div_to_matrix(divergences)
  else:
    div_mat = divergences


  if accumulate:
    div_mat = np.minimum.accumulate(div_mat)

  if fig is None:
    num_layers, num_tokens = div_mat.shape
    fig, ax = plt.subplots(figsize=(num_tokens // 5, num_layers // 5))

  heatmap = ax.imshow(div_mat, aspect='auto', vmin=0, vmax=1, cmap=cmap)
  ax.invert_yaxis()
  if tokens:
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xticks(range(len(tokens)), tokens, rotation=90)

  ax.set_ylabel('Layers')
  ax.set_xlabel('Tokens')
  fig.colorbar(heatmap)

  ax.set_title(title)

  return fig
