"""Train a CNN on a folder-based plant leaf dataset.

Expected folder structure:
dataset/
  train/
    Apple___Apple_scab/
    Apple___healthy/
    ...
  valid/              (optional; if absent, a validation split is taken from train)
    Apple___Apple_scab/
    Apple___healthy/
    ...
  test/               (optional; not used for training)

Download the PlantVillage dataset and arrange/extract it into this structure.
The script uses transfer learning with MobileNetV2 to make training practical.
"""
import json
import argparse
from pathlib import Path

import tensorflow as tf

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset", help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--steps-per-epoch", type=int, help="Limit training batches per epoch")
    args = parser.parse_args()
    if args.steps_per_epoch is not None and args.steps_per_epoch < 1:
        raise SystemExit("--steps-per-epoch must be a positive integer.")

    root = Path(args.data)
    train_dir = root / "train"
    valid_dir = root / "valid"
    if not train_dir.exists():
        raise SystemExit(f"Missing {train_dir}. See README.md for expected dataset layout.")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMAGE_SIZE, batch_size=args.batch_size,
        seed=SEED, validation_split=0.15 if not valid_dir.exists() else None,
        subset="training" if not valid_dir.exists() else None
    )
    if valid_dir.exists():
        val_ds = tf.keras.utils.image_dataset_from_directory(
            valid_dir, image_size=IMAGE_SIZE, batch_size=args.batch_size,
            shuffle=False
        )
    else:
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir, image_size=IMAGE_SIZE, batch_size=args.batch_size,
            seed=SEED, validation_split=0.15, subset="validation"
        )

    class_names = train_ds.class_names
    if valid_dir.exists() and val_ds.class_names != class_names:
        raise SystemExit("Class folders in train/ and valid/ must match exactly.")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    train_batches = int(tf.data.experimental.cardinality(train_ds).numpy())
    if args.steps_per_epoch is not None and args.steps_per_epoch > train_batches:
        raise SystemExit(
            f"--steps-per-epoch cannot exceed the {train_batches} available training batches."
        )

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomContrast(0.1)
    ], name="augmentation")

    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = augmentation(inputs)
    # MobileNetV2 expects inputs scaled to [-1, 1].
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.ModelCheckpoint("models/best_checkpoint.keras", save_best_only=True, monitor="val_accuracy")
    ]
    Path("models").mkdir(exist_ok=True)
    print(f"Training {len(class_names)} classes: {class_names}")
    model.fit(
        train_ds, validation_data=val_ds, epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch, callbacks=callbacks
    )

    if args.fine_tune_epochs > 0:
        base.trainable = True
        # Fine-tune only the last layers for stability.
        for layer in base.layers[:-30]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.fit(
            train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
            steps_per_epoch=args.steps_per_epoch, callbacks=callbacks
        )

    model.save("models/plant_disease_model.keras")
    Path("models/class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print("Saved models/plant_disease_model.keras and models/class_names.json")
    print("Next: run python app.py and upload a leaf image.")

if __name__ == "__main__":
    main()
