import numpy as np
from torchvision.datasets import CocoDetection
from tqdm import tqdm


DOG = 18


def has_dog(annotations):
    for annotation in annotations:
        if annotation['category_id'] == DOG:
            return True
    return False


def shuffle_pair(images, labels):
    indices = np.random.permutation(len(images))
    return images[indices], labels[indices]


def get_dataset(dataset):
    try:
        train_images = np.load('train_images.npy')
        train_labels = np.load('train_labels.npy')
        test_images = np.load('test_images.npy')
        test_labels = np.load('test_labels.npy')
    except FileNotFoundError:
        dogs = []
        not_dogs = []
        for idx, (_, annotations) in enumerate(tqdm(dataset)):
            if has_dog(annotations):
                dogs.append(idx)
            else:
                not_dogs.append(idx)
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

        np.save('train_images.npy', train_images)
        np.save('train_labels.npy', train_labels)
        np.save('test_images.npy', test_images)
        np.save('test_labels.npy', test_labels)

    print('Train size:', len(train_images))
    print('Test size:', len(test_images))


if __name__ == '__main__':
    dataset = CocoDetection(root='train2017', annFile='annotations/instances_train2017.json')
    get_dataset(dataset)
