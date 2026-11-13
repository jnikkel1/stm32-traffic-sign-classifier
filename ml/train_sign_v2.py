"""
train_sign_v2.py  -  GTSRB trainer for STM32F446RE (X-CUBE-AI, int8)

What changed versus signtraining.py / train_gtsrb_tiny_cnn.py, and why:

 1. Receptive field. The old nets were ~19-22 px wide at the last conv layer
    and then averaged globally, so each feature only ever saw about a sixth of
    the sign. That is enough for "red octagon" / "red triangle" / "yellow
    diamond" but not for digits and pictograms. This net downsamples to 3x3,
    where every feature sees the whole 48x48 input, and keeps spatial layout
    with Flatten instead of global average pooling.
 2. Class weights are actually passed to fit() (they were computed but unused).
 3. Validation split is by physical sign (GTSRB "track"), not by random image.
    Random splits put near-identical frames on both sides and inflate accuracy.
 4. Camera-like augmentation, applied per image: white-balance error, brightness,
    contrast, saturation, blur/low resolution, noise, RGB565 bit depth.
 5. The model takes RAW 0..255 pixels (no /255 anywhere). On the chip the int8
    input is then simply  raw_byte - 128  (verify scale/zero_point printed below).
 6. Reports per-class recall on the official Test set for float AND int8, plus an
    int8 "camera-degraded" stress test, so you can see which signs fail.

Dataset layout (Kaggle GTSRB):  <data_dir>/Train/<0..42>/<track>_<frame>.png,
                                <data_dir>/Test/*.png,  <data_dir>/Test.csv

Usage:  python train_sign_v2.py --data_dir GTSRB --epochs 60
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers, models, regularizers

IMG_SIZE = 48          # keep equal to the firmware's MODEL_IMG_W/H
NUM_CLASSES = 43
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def split_by_track(train_dir, val_frac=0.15):
    """File names are <class>_<track>_<frame>.png; one track = one physical sign seen in
    30 consecutive frames. Whole tracks go to train OR val, never both."""
    rng = random.Random(SEED)
    tr_p, tr_y, va_p, va_y = [], [], [], []
    for c in range(NUM_CLASSES):
        cdir = os.path.join(train_dir, str(c))
        if not os.path.isdir(cdir):
            print(f"  warning: missing folder {cdir}")
            continue
        files = sorted(f for f in os.listdir(cdir) if f.lower().endswith(".png"))
        
        # Changed index from [0] to [1] to grab the track ID from the middle of the filename
        tracks = sorted({f.split("_")[1] for f in files}) 
        
        rng.shuffle(tracks)
        n_val = max(1, round(len(tracks) * val_frac)) if len(tracks) > 1 else 0
        val_tracks = set(tracks[:n_val])
        for f in files:
            p = os.path.join(cdir, f)
            # Changed index here as well
            if f.split("_")[1] in val_tracks:
                va_p.append(p); va_y.append(c)
            else:
                tr_p.append(p); tr_y.append(c)
    return tr_p, np.array(tr_y, np.int64), va_p, np.array(va_y, np.int64)

def load_image(path):
    img = tf.io.decode_png(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0.0, 255.0), tf.uint8)


def load_test(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "Test.csv"))
    imgs = np.stack([load_image(os.path.join(data_dir, p)).numpy() for p in df["Path"]])
    return imgs, df["ClassId"].to_numpy(np.int64)


def photometric(img):
    """img: float32 [H, W, 3], raw 0..255. Roughly what the OV7670 does to a scene."""
    img = img * tf.random.uniform([1, 1, 3], 0.80, 1.20)                  # white-balance error
    img = tf.image.random_brightness(img, 30.0)
    img = tf.image.random_contrast(img, 0.7, 1.3)
    gray = tf.reduce_mean(img, axis=-1, keepdims=True)
    img = gray + (img - gray) * tf.random.uniform([], 0.6, 1.4)            # saturation
    s = tf.random.uniform([], 24, IMG_SIZE + 1, dtype=tf.int32)            # soft optics / low res
    img = tf.image.resize(tf.image.resize(img, [s, s], method="area"),
                          [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = img + tf.random.normal(tf.shape(img), stddev=4.0)                # sensor noise
    img = tf.concat([tf.floor(img[..., 0:1] / 8.0) * 8.0,                  # RGB565 bit depth
                     tf.floor(img[..., 1:2] / 4.0) * 4.0,
                     tf.floor(img[..., 2:3] / 8.0) * 8.0], axis=-1)
    return tf.clip_by_value(img, 0.0, 255.0)


geometric = tf.keras.Sequential([
    layers.RandomRotation(0.05, fill_mode="nearest"),
    layers.RandomZoom(0.2, fill_mode="nearest"),
    layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
])


def make_ds(paths, labels, batch, train):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (load_image(p), y), num_parallel_calls=AUTOTUNE).cache()  # uint8 cache
    if train:
        ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(lambda x, y: (photometric(tf.cast(x, tf.float32)), y),
                    num_parallel_calls=AUTOTUNE)                           # per image
        ds = ds.batch(batch).map(lambda x, y: (geometric(x, training=True), y),
                                 num_parallel_calls=AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y),
                    num_parallel_calls=AUTOTUNE).batch(batch)
    return ds.prefetch(AUTOTUNE)


def class_weights(labels):
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
    w = (counts.mean() / np.maximum(counts, 1.0)) ** 0.5      # sqrt-balanced: helps rare
    return {i: float(w[i]) for i in range(NUM_CLASSES)}       # classes without destabilising


# --------------------------------------------------------------------------- #
# Model: ~150k params (~150 KB int8), peak activations ~25 KB int8, ~8M MACs.
# Only Conv2D + BN + ReLU6 + Flatten + Dense + Softmax (all X-CUBE-AI friendly).
# --------------------------------------------------------------------------- #
def conv_bn(x, filters, stride):
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False,
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU(max_value=6.0)(x)


def build_model():
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="raw_rgb")  # raw 0..255, no rescaling
    x = conv_bn(inp, 16, 2)     # 24x24
    x = conv_bn(x, 24, 1)
    x = conv_bn(x, 32, 2)       # 12x12
    x = conv_bn(x, 48, 1)
    x = conv_bn(x, 64, 2)       # 6x6
    x = conv_bn(x, 64, 1)
    x = conv_bn(x, 64, 2)       # 3x3: receptive field now covers the whole input
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return models.Model(inp, out, name="sign_net_v2")


# --------------------------------------------------------------------------- #
# int8 export + evaluation
# --------------------------------------------------------------------------- #
def export_int8(model, calib_ds, path, n_calib=300):
    def representative_dataset():
        # clean image AND a camera-degraded copy, so activation ranges cover both
        for x, _ in calib_ds.unbatch().shuffle(4096, seed=SEED).batch(1).take(n_calib):
            yield [x]
            yield [tf.expand_dims(photometric(x[0]), 0)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative_dataset
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    blob = conv.convert()
    with open(path, "wb") as f:
        f.write(blob)
    print(f"\nSaved {path}  ({len(blob) / 1024:.1f} KB)")


def make_interpreter(path):
    try:
        it = tf.lite.Interpreter(model_path=path)
    except AttributeError:                       # newer TF removed tf.lite.Interpreter
        from ai_edge_litert.interpreter import Interpreter
        it = Interpreter(model_path=path)
    it.allocate_tensors()
    return it


def predict_int8(path, x_float):
    it = make_interpreter(path)
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]
    scale, zp = ind["quantization"]
    preds = np.empty(len(x_float), np.int64)
    for i in range(len(x_float)):
        q = np.clip(np.round(x_float[i:i + 1] / scale + zp), -128, 127).astype(np.int8)
        it.set_tensor(ind["index"], q)
        it.invoke()
        preds[i] = int(np.argmax(it.get_tensor(outd["index"])[0]))
    return preds


def print_quant_info(path):
    it = make_interpreter(path)
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]
    print("Quantization (needed in firmware):")
    print(f"  input : shape={ind['shape'].tolist()} scale={ind['quantization'][0]:.6f} "
          f"zero_point={ind['quantization'][1]}   (scale~1, zp~-128  =>  int8 = raw_byte - 128)")
    print(f"  output: scale={outd['quantization'][0]:.6f} zero_point={outd['quantization'][1]}")


def report(name, y_true, y_pred, csv_path=None):
    acc = float((y_true == y_pred).mean())
    recall = np.array([(y_pred[y_true == c] == c).mean() if (y_true == c).any() else np.nan
                       for c in range(NUM_CLASSES)])
    weakest = np.argsort(np.nan_to_num(recall, nan=2.0))[:10]
    print(f"\n{name}: {acc * 100:.2f}%")
    print(f"  classes below 50% recall: {int(np.nansum(recall < 0.5))} of {NUM_CLASSES}")
    print("  weakest (class: recall): " + ", ".join(f"{c}: {recall[c] * 100:.0f}%" for c in weakest))
    if csv_path:
        pd.DataFrame({"class": range(NUM_CLASSES), "recall": recall}).to_csv(csv_path, index=False)
    return acc


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="GTSRB")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", default="sign_v2")
    args = ap.parse_args()

    print("Splitting Train by track...")
    tr_p, tr_y, va_p, va_y = split_by_track(os.path.join(args.data_dir, "Train"))
    print(f"  train: {len(tr_p)} images | val: {len(va_p)} images (different physical signs)")
    print("  images per class (train):", np.bincount(tr_y, minlength=NUM_CLASSES).tolist())

    train_ds = make_ds(tr_p, tr_y, args.batch_size, train=True)
    val_ds = make_ds(va_p, va_y, args.batch_size, train=False)

    model = build_model()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        class_weight=class_weights(tr_y),
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5),
        ],
    )
    model.save(f"{args.out}.keras")

    print("\nLoading official Test set...")
    x_test, y_test = load_test(args.data_dir)
    x_test = x_test.astype(np.float32)
    p_float = np.argmax(model.predict(x_test, batch_size=256, verbose=0), axis=1)
    report("float32  | official Test set", y_test, p_float)

    tflite_path = f"{args.out}_int8.tflite"
    export_int8(model, make_ds(tr_p, tr_y, args.batch_size, train=False), tflite_path)
    print_quant_info(tflite_path)

    report("int8     | official Test set", y_test, predict_int8(tflite_path, x_test),
           csv_path=f"{args.out}_per_class_recall.csv")

    tf.random.set_seed(123)
    x_stress = np.stack([photometric(tf.constant(x)).numpy() for x in x_test])
    report("int8     | Test set + camera-like degradation", y_test,
           predict_int8(tflite_path, x_stress))
    print("\nThe gap between the last two numbers is how fragile the model is to camera-like "
          "image quality. Re-import the .tflite in CubeMX and check the Analyze report.")


if __name__ == "__main__":
    main()
