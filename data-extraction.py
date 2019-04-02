# Based on:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

import os
import gzip
import numpy as np

def get_first_file(path):
    filename = os.listdir(path)[0]
    return os.path.join(path, filename)


def _load_images(file):
    """
    :param file: A file object that can be passed into a gzip reader.
    :return: A 4D uint8 numpy array [index -> y -> x -> brightness]
    :raises ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting {}'.format(file.name))
    with gzip.GzipFile(fileobj=file) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' % (magic, file.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def _load_labels(file):
    """
    :param file: A file object that can be passed into a gzip reader.
    :return: A 1D uint8 numpy array [index -> label between 0-9]
    :raises ValueError: If the bytestream doesn't start with 2049.
    """
    print('Extracting {}'.format(file.name))
    with gzip.GzipFile(fileobj=file) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' % (magic, file.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def _save_images(output_file_path, image_data):
    """
    :param output_file_path: Where to save the archive, filename included.
    :param image_data: the 4D np array to save
    :return:
    """
    header = np.array([0x0803, len(image_data), 28, 28], dtype='>i4')
    with gzip.open(output_file_path, "wb") as f:
        f.write(header.tobytes())
        f.write(image_data.tobytes())


def _save_labels(output_file_path, label_data):
    """
    :param output_file_path: Where to save the archive, filename included.
    :param label_data: the numpy array to save.
    :return:
    """
    header = np.array([0x0801, len(label_data)], dtype='>i4')
    with gzip.open(output_file_path, "wb") as f:
        f.write(header.tobytes())
        f.write(label_data.tobytes())


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def main():
    INPUTS_DIR = os.getenv('VH_INPUTS_DIR', './inputs')
    DATA_DIR = os.path.join(INPUTS_DIR, 'dataset')

    data_path = get_first_file(DATA_DIR)

    with open(data_path, 'rb') as f:
        data = _load_images(f)

    # Note that this is only for demoing purposes, only extracts and compresses the data.
    print('Applying feature extraction...')
    # TODO: For the sake of example, use actual images to generate features?
    # TODO: Add parameters to control the feature extraction?

    OUTPUTS_DIR = os.getenv('VH_OUTPUTS_DIR', './outputs')
    _save_images(os.path.join(OUTPUTS_DIR, 'data.gz'), data)

if __name__ == '__main__':
    main()