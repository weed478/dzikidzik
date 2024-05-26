import keras
import tensorflow as tf
import numpy as np


def build_model():
    backbone = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        pooling='avg',
        weights='imagenet',
        alpha=0.35,
    )
    model = keras.Sequential([
        keras.Input(shape=(224, 224, 3)),
        backbone,
        keras.layers.Dense(1, activation='sigmoid'),
    ])
    return model


def export(model):
    saved_model_dir = 'doggy_saved_model'
    model.export(saved_model_dir)

    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, 224, 224, 3) * 2 - 1
            yield [data.astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()

    with open('doggy.tflite', 'wb') as f:
        f.write(tflite_quant_model)


def main():
    train_images = np.load('coco/train_images.npy')
    train_labels = np.load('coco/train_labels.npy')
    test_images = np.load('coco/test_images.npy')
    test_labels = np.load('coco/test_labels.npy')

    model = build_model()

    epochs = 5
    batch_size = 32

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=1e-4,
                decay_steps=epochs * len(train_images) // batch_size,
            ),
            clipnorm=1.0,
        ),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.AUC()],
    )

    model.fit(
        train_images, train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(test_images, test_labels),
    )

    model.save('doggy.keras')

    export(model)


if __name__ == '__main__':
    main()
