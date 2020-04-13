import os
from flax import serialization
from tensorflow.compat.v2.io import gfile


def save_model(filename, model):
    gfile.makedirs(os.path.dirname(filename))
    with gfile.GFile(filename, 'wb') as fp:
        fp.write(serialization.to_bytes(model))


def load_model(filename, model):
    with gfile.GFile(filename, 'rb') as fp:
        return serialization.from_bytes(model, fp.read())
