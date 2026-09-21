"""DS-CNN keyword-spotting architecture (depthwise-separable CNN).

Reference: Zhang et al., "Hello Edge: Keyword Spotting on Microcontrollers"
(arXiv:1711.07128).  DS-CNN is the shape that won the MCU accuracy-per-byte
trade-off and it is still the right answer when the budget is 256 KB of RAM:

* depthwise 3x3 convolutions do almost all of the spatial work at 1/9th the
  MACs of a dense 3x3 convolution,
* 1x1 pointwise convolutions mix channels at 1 MAC per element,
* global average pooling before the output removes the fat dense head that
  dominates parameter counts in naive CNNs,
* downsampling happens exactly once per selected block (strided depthwise conv),
  so the encoder keeps a 7-step time axis (49 -> 25 -> 13 -> 7),
* every op has an int8 TFLite equivalent, so full-integer quantisation leaves
  **zero** float fallback ops (verified in ``ml/tools/export_tflite.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras import layers


@dataclass
class DSCNNConfig:
    """Shape and capacity knobs.  ``width`` scales every channel count."""

    input_frames: int = 49
    input_bins: int = 40
    width: float = 1.0
    stem_filters: int = 64
    stem_kernel: tuple[int, int] = (10, 4)
    stem_stride: tuple[int, int] = (2, 2)
    block_filters: tuple[int, ...] = (64, 64, 64, 128, 128, 128)
    #: blocks whose depthwise conv uses stride 2 (halves both time and frequency)
    stride_after: tuple[int, ...] = (3, 5)
    dropout: float = 0.2
    l2: float = 1e-5
    #: ``single`` -> one sigmoid output (production).  ``multi`` -> an auxiliary
    #: per-frame head used only during training; see :func:`build_multitask_dscnn`.
    head: str = "single"


def _scaled(n: int, width: float) -> int:
    return max(8, int(round(n * width / 8.0) * 8))


def build_dscnn(cfg: DSCNNConfig | None = None) -> tf.keras.Model:
    """Single-logit streaming detector.

    The model answers one question per 0.98 s analysis window: "did the keyword
    end within this window?".  ``sigmoid = 1`` therefore marks a *keyword end
    event*, which is what makes the latency measurable: the detection fires as
    soon as the last phoneme of the keyword has entered the window.
    """
    cfg = cfg or DSCNNConfig()
    reg = tf.keras.regularizers.l2(cfg.l2)

    inp = layers.Input(shape=(cfg.input_frames, cfg.input_bins, 1), name="logmel_int8")
    x = layers.Conv2D(
        _scaled(cfg.stem_filters, cfg.width),
        cfg.stem_kernel,
        strides=cfg.stem_stride,
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(max_value=6.0, name="stem_relu")(x)  # ReLU6: int8-friendly

    for i, filters in enumerate(cfg.block_filters):
        f = _scaled(filters, cfg.width)
        # Downsampling happens exactly ONCE per selected block, via the stride of
        # the depthwise conv.  Adding a MaxPool2D on top of a stride-2 conv (an
        # easy mistake) quarters the time axis per block and collapses the
        # encoder to 2 time steps, which destroys the frame-level supervision.
        stride = (2, 2) if i in cfg.stride_after else (1, 1)
        x = layers.DepthwiseConv2D(
            3, strides=stride, padding="same", depth_multiplier=1, use_bias=False, name=f"dw{i}"
        )(x)
        x = layers.BatchNormalization(name=f"dw{i}_bn")(x)
        x = layers.ReLU(max_value=6.0, name=f"dw{i}_relu")(x)
        x = layers.Conv2D(f, 1, strides=1, padding="same", use_bias=False, name=f"pw{i}")(x)
        x = layers.BatchNormalization(name=f"pw{i}_bn")(x)
        x = layers.ReLU(max_value=6.0, name=f"pw{i}_relu")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    if cfg.dropout > 0:
        x = layers.Dropout(cfg.dropout, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", kernel_regularizer=reg, name="wake")(x)
    return tf.keras.Model(inp, out, name=f"dscnn_w{int(cfg.width * 100)}")


def build_multitask_dscnn(cfg: DSCNNConfig | None = None) -> tf.keras.Model:
    """DS-CNN with two heads, used only for training.

    * ``wake``     - the deployed window-level head (identical to
      :func:`build_dscnn`, so its weights transfer unchanged).
    * ``frame``    - an auxiliary head that predicts, for every time step of the
      *encoder output*, whether the keyword is currently in progress.  Frame
      supervision is what teaches the encoder to be position-aware instead of
      collapsing into "the keyword is somewhere in these 0.98 s", which in turn
      sharpens the wake-up edge and cuts latency.  This head is stripped on
      export.

    Note the architecture is fully convolutional in time, so the same weights
    can also be evaluated on longer windows - useful for the "second opinion"
    re-scoring described in ``docs/ARCHITECTURE.md``.
    """
    cfg = cfg or DSCNNConfig()
    cfg = DSCNNConfig(**{**cfg.__dict__, "head": "multi"})
    reg = tf.keras.regularizers.l2(cfg.l2)

    inp = layers.Input(shape=(cfg.input_frames, cfg.input_bins, 1), name="logmel_int8")
    x = layers.Conv2D(
        _scaled(cfg.stem_filters, cfg.width), cfg.stem_kernel, strides=cfg.stem_stride,
        padding="same", use_bias=False, name="stem_conv",
    )(inp)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(max_value=6.0, name="stem_relu")(x)

    for i, filters in enumerate(cfg.block_filters):
        f = _scaled(filters, cfg.width)
        stride = (2, 2) if i in cfg.stride_after else (1, 1)
        x = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False, name=f"dw{i}")(x)
        x = layers.BatchNormalization(name=f"dw{i}_bn")(x)
        x = layers.ReLU(max_value=6.0, name=f"dw{i}_relu")(x)
        x = layers.Conv2D(f, 1, strides=1, padding="same", use_bias=False, name=f"pw{i}")(x)
        x = layers.BatchNormalization(name=f"pw{i}_bn")(x)
        x = layers.ReLU(max_value=6.0, name=f"pw{i}_relu")(x)

    # ---- window head (this is the part that ships) ----------------------
    g = layers.GlobalAveragePooling2D(name="gap")(x)
    if cfg.dropout > 0:
        g = layers.Dropout(cfg.dropout, name="dropout")(g)
    wake = layers.Dense(1, activation="sigmoid", kernel_regularizer=reg, name="wake")(g)

    # ---- auxiliary frame head (training only) --------------------------
    # Frequency-average the encoder output to get a T'-step temporal feature
    # map, then run a small temporal convolver that still sees enough context
    # to decide "keyword in progress" (the keyword is ~0.6 s ~ 15 frames).
    tf_feat = layers.Lambda(lambda t: tf.reduce_mean(t, axis=2), name="freq_pool")(x)  # (B, T', C)
    # Kernel size 3, not 11: the encoder output is only 7 time steps long, so a
    # wider kernel would convolve mostly padding and would feed the shared encoder
    # noise instead of signal.
    t = layers.Conv1D(64, 3, padding="same", use_bias=False, name="frame_conv1")(tf_feat)
    t = layers.BatchNormalization(name="frame_bn1")(t)
    t = layers.ReLU(max_value=6.0, name="frame_relu1")(t)
    t = layers.Conv1D(32, 3, padding="same", use_bias=False, name="frame_conv2")(t)
    t = layers.BatchNormalization(name="frame_bn2")(t)
    t = layers.ReLU(max_value=6.0, name="frame_relu2")(t)
    frame = layers.Dense(1, activation="sigmoid", kernel_regularizer=reg, name="frame")(t)  # (B, T', 1)

    return tf.keras.Model(inp, {"wake": wake, "frame": frame}, name=f"dscnn_mt_w{int(cfg.width * 100)}")


def count_parameters(model: tf.keras.Model) -> int:
    """Trainable + non-trainable parameter count (a proxy for flash footprint)."""
    return int(sum(int(tf.size(w)) for w in model.weights))


def _shape_of(t) -> list:
    """Keras 3 exposes ``model.inputs``/``outputs`` as tuples of tensors."""
    return list(getattr(t, "shape", ()) or ())


def describe(model: tf.keras.Model) -> dict:
    """Small summary used in the model card."""
    layers_out = []
    for layer in model.layers:
        try:
            out = layer.output
            if isinstance(out, (list, tuple)):
                shape = [_shape_of(t) for t in out]
            elif hasattr(out, "shape"):
                shape = _shape_of(out)
            else:
                shape = None
        except Exception:
            shape = None
        layers_out.append(
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "params": int(layer.count_params()),
                "output": shape,
            }
        )
    return {
        "name": model.name,
        "inputs": [_shape_of(t) for t in model.inputs],
        "outputs": [_shape_of(t) for t in model.outputs],
        "params_total": count_parameters(model),
        "layers": layers_out,
    }
