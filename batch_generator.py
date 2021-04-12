import numpy as np
from dataset import load_data

class BatchGenerator:
    TRAIN = 1
    TEST = 0

    def __init__(self, data_src, seed, batch_size=32, dataset='MNIST'):
        self.batch_size = batch_size
        self.data_src = data_src

        # Load data
        ((x, y), (x_test, y_test)) = load_data(dataset,
                                               seed=seed,
                                               imbalance=True)  # tf.keras.datasets.cifar10.load_data()

        if self.data_src == self.TRAIN:
            self.dataset_x = x
            self.dataset_y = y
        else:
            self.dataset_x = x_test
            self.dataset_y = y_test

        # Arrange x: channel first
        self.dataset_x = np.transpose(self.dataset_x, axes=(0, 3, 1, 2))

        # Normalize between -1 and 1
        # self.dataset_x = self.dataset_x / 255 - 0.5

        # Y 1D format
        # self.dataset_y = self.dataset_y[:, 0]

        assert (self.dataset_x.shape[0] == self.dataset_y.shape[0])

        # Compute per class instance count.
        classes = np.unique(self.dataset_y)
        self.classes = classes
        per_class_count = list()
        for c in classes:
            per_class_count.append(np.sum(np.array(self.dataset_y == c)))

        # Recount after pruning
        per_class_count = list()
        for c in classes:
            per_class_count.append(np.sum(np.array(self.dataset_y == c)))
        self.per_class_count = per_class_count

        # List of labels
        self.label_table = [str(c) for c in range(len(self.classes))]

        # Preload all the labels.
        self.labels = self.dataset_y[:]

        # per class ids
        self.per_class_ids = dict()
        ids = np.array(range(len(self.dataset_x)))
        for c in classes:
            self.per_class_ids[c] = ids[self.labels == c]

    def get_samples_for_class(self, c, samples=None):
        if samples is None:
            samples = self.batch_size

        np.random.shuffle(self.per_class_ids[c])
        to_return = self.per_class_ids[c][0:samples]
        return self.dataset_x[to_return]

    def get_label_table(self):
        return self.label_table

    def get_num_classes(self):
        return len(self.label_table)

    def get_class_probability(self):
        return self.per_class_count / sum(self.per_class_count)

    ### ACCESS DATA AND SHAPES ###
    def get_num_samples(self):
        return self.dataset_x.shape[0]

    def get_image_shape(self):
        return [self.dataset_x.shape[1], self.dataset_x.shape[2], self.dataset_x.shape[3]]

    def next_batch(self):
        dataset_x = self.dataset_x
        labels = self.labels

        indices = np.arange(dataset_x.shape[0])

        # np.random.shuffle(indices)

        for start_idx in range(0, dataset_x.shape[0], self.batch_size):
            access_pattern = indices[start_idx:start_idx + self.batch_size]
            access_pattern = sorted(access_pattern)

            yield dataset_x[access_pattern, :, :, :], labels[access_pattern]
