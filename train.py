import keras
import tensorflow as tf
import numpy as np
from torchvision.datasets import CocoDetection
from tqdm import tqdm


DOG = 18


def has_dog(annotations):
    for annotation in annotations:
        if annotation['category_id'] == DOG:
            return True
    return False


def preprocess_image(image):
    return np.array(image.resize((224, 224))).astype(np.float32) / 127.5 - 1


def shuffle_pair(images, labels):
    indices = np.random.permutation(len(images))
    return images[indices], labels[indices]


def get_dataset(dataset):
    dogs = []
    not_dogs = []
    for image, annotations in tqdm(dataset):
        image = preprocess_image(image)
        if has_dog(annotations):
            dogs.append(image)
        else:
            not_dogs.append(image)
    images = np.array(dogs + not_dogs)
    np.random.shuffle(images)

    test_size = len(dogs) // 5
    test_dogs = dogs[:test_size // 2]
    test_not_dogs = not_dogs[:test_size // 2]
    test_images = np.array(test_dogs + test_not_dogs)
    test_labels = np.array([1] * len(test_dogs) + [0] * len(test_not_dogs))
    test_images, test_labels = shuffle_pair(test_images, test_labels)

    train_dogs = dogs[test_size // 2:]
    train_not_dogs = not_dogs[test_size // 2:len(dogs)]
    train_images = np.array(train_dogs + train_not_dogs)
    train_labels = np.array([1] * len(train_dogs) + [0] * len(train_not_dogs))
    train_images, train_labels = shuffle_pair(train_images, train_labels)

    return (
        tf.data.Dataset.from_tensor_slices((train_images, train_labels)),
        tf.data.Dataset.from_tensor_slices((test_images, test_labels)),
    )


def build_model():
    backbone = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        pooling='avg',
        weights='imagenet',
        alpha=0.5,
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
    dataset = CocoDetection(root='coco/val2017', annFile='coco/annotations/instances_val2017.json')
    train_dataset, test_dataset = get_dataset(dataset)

    model = build_model()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(), keras.metrics.AUC()],
    )

    model.fit(
        train_dataset.batch(32).prefetch(tf.data.AUTOTUNE),
        epochs=10,
        validation_data=test_dataset.batch(32).prefetch(tf.data.AUTOTUNE),
    )

    model.save('doggy.keras')

    export(model)


if __name__ == '__main__':
    main()
