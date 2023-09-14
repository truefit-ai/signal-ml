##  *signal-ml*

#### Optimized transformers for signal processing

It may not be the best package, but it is the **only** package. 

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

## Architectures:
 - Vision Transformer
 - Vision Transformer (Relative Position)
 - Vision Transformer (Alibi)
 - Disentangled Self-Attention (aka 'deberta')

*(they're all good architectures)*

&nbsp;

## Parameters

#### Key Parameters
 - encoder: vt, vt-relpos, vt-alibi, deberta, None [default: vt-relpos]
 - pooling: gru, lstm, gem, avg, cls, None [default: gru]
 - patch: 1 - ∞ [default: 4]

*(ideally gru/gem; patch ~= seqlen // 384)*

&nbsp;

#### Data Parameters
 - in_channels: 1 - ∞ [default: 1]
 - out_channels: 1 - ∞ or None [default: None]
 - out_act: None, sigmoid, tanh, softmax [default: None]

*(match your dataset!)*

&nbsp;

#### Power Parameters
 - depth: 0 - ∞ [default: 3]
 - dim: 64 - ∞ [default: 256]
 - multi_sample: 1 - ∞ [default: 10]
 - num_heads: 1 - ∞ [default: 4]
 - act_layer: gelu, celu, silu, prelu [default: gelu]

*(dim=384 is slightly better; usually depth 2-5)*

&nbsp;

#### Regularization 
 - drop_rate: 0 - 1 [default: 0.2]
 - proj_drop_rate: 0 - 1 [default: 0.1]
 - attn_drop_rate: 0 - 1 [default: 0.1]
 - drop_path_rate: 0 - 1 [default: 0.0]
 - init_values: 0 - 1 [default: 0.15]

*(attn_drop_rate can be zero; others are optimal)*

&nbsp;

### Automatic Loss Functions
 - loss: mse, ce, None [default: None]
 - smooth: 0 - 1 [default: 3e-4]
 - ae_loss: 0 - 1 [default: 0.03]

*(matches your objective)

&nbsp;

### Class Outputs

 - pred: overall prediction 
 - pred_patch: patch prediction
 - pred_point: point prediction
 - loss: overall loss

*(automatic inference for all variations)

&nbsp;

## Classification:

```python
import signal_ml as sml

model = sml.SigModel(
    encoder = 'vt-relpos',
    pooling = 'gru',
    in_channels = 3,
    out_channels = 20
    out_act = 'softmax'
    loss = 'ce'
)

x = torch.randn(8, 1024, 3)
y = torch.randint(0, 20, (8, ))
yp = model(x, target = y)
print(yp.pred, yp.loss)

```

&nbsp;

## Regression 

```python
model = sml.SigModel(
    pooling = None,
    patch = 18,
    in_channels = 4,
    out_channels = 1, 
    loss = 'mse',
)

x = torch.randn(4, 3600, 4)
y = torch.randn(4, 200, 1)
yp = model(x, target = y)
print(yp.pred_patch, yp.loss)

```

&nbsp;

## Modules

Full PyTorch Lightning Modules—–with optimizer, loss function, metrics, and more 

```python
import signal_ml as sml

module = sml.SigModule(
    loss = 'ce',
    final_act = 'softmax',
    optimizer = 'adamw',
    lr = 3e-5,
    schedule = 'lincos',
    n_epochs = 20,
)

module.fit(train_loader, test_loader)

```

##### Module Parameters:
 - optimizer: adamw, lion [default: adamw]
 - lr: 0 - 1 [default: 3e-5]
 - wd: 0 - 1 [default: 0.01]
 - schedule: lincos, func [default: lincos]
 - head_mult: 1 - ∞ [default: 10]
 - n_epochs: 1 - ∞ [default: 10]

*(tune the learning rate! sometimes tune head_mult or try lion)*

&nbsp;

### Module Features
 - automatic mixed precision 
 - automatic accelerator 
 - automatic scheduler 
 - automatic metrics 
 - and more...

***

## Sample Datasets:

etc. 







