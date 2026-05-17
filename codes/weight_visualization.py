# codes to make visualization of your weights.
import math

import matplotlib.pyplot as plt
import numpy as np

import mynn as nn
from mynn.op import conv2D, Linear


def _load_model(model_path, model_type='auto'):
    if model_type == 'cnn':
        model = nn.models.Model_CNN()
    elif model_type == 'mlp':
        model = nn.models.Model_MLP()
    elif model_type == 'auto':
        import pickle
        with open(model_path, 'rb') as f:
            param_list = pickle.load(f)
        model = nn.models.Model_CNN() if isinstance(param_list[0], dict) else nn.models.Model_MLP()
    else:
        raise ValueError("model_type should be 'auto', 'cnn', or 'mlp'")

    model.load_model(model_path)
    return model


def visualize_cnn_conv_weights(model_path, layer_index=0, save_path=None, show=True):
    model = _load_model(model_path, model_type='cnn')
    conv_layers = [layer for layer in model.layers if isinstance(layer, conv2D)]
    if layer_index >= len(conv_layers):
        raise ValueError(f'CNN only has {len(conv_layers)} conv layers.')

    weights = conv_layers[layer_index].params['W']
    out_channels, in_channels, _, _ = weights.shape
    cols = min(8, out_channels)
    rows = math.ceil(out_channels / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.8 * rows))
    axes = np.asarray(axes).reshape(-1)

    for idx, ax in enumerate(axes):
        ax.axis('off')
        if idx >= out_channels:
            continue
        weight_map = weights[idx].mean(axis=0) if in_channels > 1 else weights[idx, 0]
        im = ax.imshow(weight_map, cmap='coolwarm')
        ax.set_title(f'filter {idx}', fontsize=8)

    fig.suptitle(f'CNN Conv Layer {layer_index} Filters')
    fig.tight_layout()
    fig.colorbar(im, ax=axes.tolist(), shrink=0.75)
    if save_path is not None:
        fig.savefig(save_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def visualize_linear_weight_heatmap(model_path, layer_index=0, save_path=None, show=True, model_type='auto'):
    model = _load_model(model_path, model_type=model_type)
    linear_layers = [layer for layer in model.layers if isinstance(layer, Linear)]
    if layer_index >= len(linear_layers):
        raise ValueError(f'Model only has {len(linear_layers)} linear layers.')

    weights = linear_layers[layer_index].params['W']
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(weights, aspect='auto', cmap='coolwarm')
    ax.set_title(f'Linear Layer {layer_index} Weight Heatmap')
    ax.set_xlabel('out dim')
    ax.set_ylabel('in dim')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def visualize_baseline_cnn_weights(model_path, save_prefix=None, show=True):
    conv0_path = None if save_prefix is None else f'{save_prefix}_conv0_filters.png'
    fc0_path = None if save_prefix is None else f'{save_prefix}_fc0_heatmap.png'
    visualize_cnn_conv_weights(model_path, layer_index=0, save_path=conv0_path, show=show)
    visualize_linear_weight_heatmap(model_path, layer_index=0, save_path=fc0_path, show=show, model_type='cnn')
