# -*- coding: utf-8 -*-
"""Batch generator definition."""

import cv2
import numpy as np


class ImageBatchGenerator(object):
    """Batch generator for training on general images."""

    def __init__(self, input_files, batch_size, height, width, channel=3,
                 shuffle=False, flip_h=False):
        assert batch_size > 0, batch_size
        assert channel == 3 or channel == 1, channel

        if channel == 3:
            self._imread_flag = cv2.IMREAD_COLOR
        else:
            self._imread_flag = cv2.IMREAD_GRAYSCALE

        self._input_files = input_files
        self._batch_size = batch_size
        self._height = height
        self._width = width
        self._shuffle = shuffle
        self._flip_h = flip_h

        for ifile in input_files:
            image = cv2.imread(ifile, cv2.IMREAD_UNCHANGED)
            assert isinstance(image, np.ndarray)
            assert image.shape[:2] == (
                height, width), (image.shape[:2], (height, width))
            print('verify ' + ifile)

        self._batch_generator = self.__get_batch_generator()

    def __get_batch_generator(self):
        batch = []

        while True:
            if self._shuffle:
                file_index = np.random.permutation(self.n_samples)
            else:
                file_index = range(self.n_samples)

            for idx in file_index:
                image = cv2.imread(self._input_files[idx], self._imread_flag)

                if self._flip_h:
                    if np.random.randint(2) == 0:
                        image = image[:, ::-1]

                if image.ndim == 2:
                    image = image.reshape((1,) + image.shape)
                else:
                    image = image.transpose((2, 0, 1))

                image = image.astype(np.float32)
                image = ((image / 255.) - 0.5) * 2.

                batch.append(image)

                if len(batch) == self._batch_size:
                    yield np.asarray(batch)
                    batch = []

    @property
    def n_samples(self):
        return len(self._input_files)

    def __next__(self):
        self._batch = next(self._batch_generator)
        return self._batch

    def next(self):
        return self.__next__()
