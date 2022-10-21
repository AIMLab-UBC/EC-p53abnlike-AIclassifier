import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler -
        In each mini-batch there are same number of classes. For example, if the batch size
        is 6, and we have 3 classes, there are 2 samples from each class in every mini-batch. If the batch_size is not
        dividable on te number of classes, the number of samples from the class with higher number of data would be increased.
        When the classes are imbalanced, the classes with lower number of patches would be sampled mutiple times in each epoch.
    Modified from: https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198/
    """

    def __init__(self, labels, batch_size):
        self.labels = labels
        # set of labels
        self.labels_set = list(set(self.labels))
        # index of the data corresponds to each label
        self.label_to_indices = {label: list(np.where(np.array(self.labels)==label)[0])
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}

        self.batch_size = batch_size
        self.n_classes  = len(self.labels_set)
        # n_sample is number of data in each batch from all the labels
        divide = list(map(len, np.array_split(range(self.batch_size),
                                                    self.n_classes)))
        divide = [num for _, num in sorted(zip(list(map(len,self.label_to_indices.values())), divide), reverse=True)]
        self.n_samples = {label: divide[idx] for idx, label in enumerate(self.labels_set)}
        # len_ is number of total batches
        # class_ind_ is label that corresponds to the label with maximum number of data
        data_ratio = {label: int(math.ceil(len(self.label_to_indices[label])/self.n_samples[label]))
                      for label in self.labels_set}
        self.len_ = max(data_ratio.values())
        self.class_ind_ = max(data_ratio, key=lambda k:data_ratio[k])

    def __iter__(self):
        self.count = 0
        while self.count < self.len_:
            indices = []
            for class_ in self.labels_set:
                # if it is the last mini-batch, the size of it is lower equal than batch_size
                if self.count != self.len_ - 1:
                    num_sample = self.n_samples[class_]
                else:
                    # last mini-batch
                    num_sample = len(self.label_to_indices[self.class_ind_]) - self.used_label_indices_count[self.class_ind_]
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + num_sample])

                self.used_label_indices_count[class_] += num_sample

                if self.used_label_indices_count[class_] + num_sample > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0

            yield indices
            self.count += 1

    def __len__(self):
        return self.len_
