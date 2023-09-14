##  *signal-ml*

#### Optimized transformers for signal processing

&nbsp;

## Load *signal-ml*

```python
import signal_ml as sml

sml.SigModel(
    encoder = 'vt-relpos',
    pooling = 'gru',
    in_channels = 1,
)
```

&nbsp;

## Why Signal-ML?
 - Optimized architectures
 - Optimized hyperparameters
 - Automatic patching + pooling
 - Popular metrics and losses

&nbsp;

### Architectures:
 - Vision Transformer
 - Vision Transformer (Relative Position)
 - Vision Transformer (Alibi)
 - Disentangled Self-Attention (aka 'deberta')

&nbsp;

#### Key Parameters
 - encoder: vt, vt-relpos, vt-alibi, deberta, None [default: vt-relpos]
 - pooling: gru, lstm, gem, avg, cls, None [default: gru]
 - patch: 1 - ∞ [default: 4]

&nbsp;

#### Data Parameters
 - in_channels: 1 - ∞ [default: 1]
 - out_channels: 1 - ∞ or None [default: None]
 - out_act: None, sigmoid, tanh, softmax [default: None]

&nbsp;

#### Power Parameters
 - depth: 0 - ∞ [default: 3]
 - encoder_dim: 64 - ∞ [default: 256]
 - multi_sample: 1 - ∞ [default: 10]
 - num_heads: 1 - ∞ [default: 4]
 - act_layer: gelu, celu, silu, prelu [default: gelu]

&nbsp;

#### Regularization 
 - drop_rate: 0 - 1 [default: 0.2]
 - proj_drop_rate: 0 - 1 [default: 0.1]
 - attn_drop_rate: 0 - 1 [default: 0.1]
 - drop_path_rate: 0 - 1 [default: 0.0]
 - init_values: 0 - 1 [default: 0.15]

&nbsp;

### Automatic Loss Functions
 - loss: mse, ce, None [default: None]
 - ae_loss: 0 - 1 [default: 0.03]



