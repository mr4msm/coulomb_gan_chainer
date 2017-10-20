# -*- coding: utf-8 -*-

import imp
import os
from chainer import functions as F
from chainer import serializers


def load_module(module_path):
    """Load Python module."""
    head, tail = os.path.split(module_path)
    module_name = os.path.splitext(tail)[0]
    info = imp.find_module(module_name, [head])
    return imp.load_module(module_name, *info)


def initialize_model(model, param=None, output_path=None):
    """Save initial params or load params to resume."""
    if param is None:
        if output_path is not None:
            serializers.save_npz(output_path, model)
            print('save ' + output_path)
    else:
        ext = os.path.splitext(param)[1]

        if ext == '.npz':
            load_func = serializers.load_npz
        elif ext == '.h5':
            load_func = serializers.load_hdf5
        else:
            raise TypeError(
                'The format of \"{}\" is not supported.'.format(param))

        load_func(param, model)
        print('load ' + param)


def initialize_optimizer(optimizer, state=None, output_path=None):
    """Save initial state or load state to resume."""
    initialize_model(optimizer, state, output_path)


def l2_norm(var):
    if var.data.ndim > 1:
        return F.sqrt(F.sum(var * var,
                            axis=tuple(range(1, var.data.ndim))))
    else:
        return var
