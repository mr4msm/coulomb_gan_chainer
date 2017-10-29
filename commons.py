# -*- coding: utf-8 -*-

import chainer
import imp
import numpy as np
import os
from chainer import functions as F
from chainer import serializers


class ModelOptimizerSet(object):
    """A set of a nn model and its optimizer."""

    SAVE_PARAM_FORMAT = 'trained-params_{0}_update-{1:09d}.npz'
    SAVE_STATE_FORMAT = 'optimizer-state_{0}_update-{1:09d}.npz'

    def __init__(self, model, optimizer):
        assert isinstance(model, chainer.Link)
        assert isinstance(optimizer, chainer.Optimizer)
        self.model = model
        self.optimizer = optimizer

    def save_model(self, model_type, out_dir='./'):
        output_file_path = os.path.join(
            out_dir,
            ModelOptimizerSet.SAVE_PARAM_FORMAT.format(model_type,
                                                       self.optimizer.t))
        serializers.save_npz(output_file_path, self.model)
        print('save ' + output_file_path)

    def save_optimizer(self, model_type, out_dir='./'):
        output_file_path = os.path.join(
            out_dir,
            ModelOptimizerSet.SAVE_STATE_FORMAT.format(model_type,
                                                       self.optimizer.t))
        serializers.save_npz(output_file_path, self.optimizer)
        print('save ' + output_file_path)

    def save(self, model_type, out_dir='./'):
        self.save_model(model_type, out_dir=out_dir)
        self.save_optimizer(model_type, out_dir=out_dir)


def load_module(module_path):
    """Load Python module."""
    head, tail = os.path.split(module_path)
    module_name = os.path.splitext(tail)[0]
    info = imp.find_module(module_name, [head])
    return imp.load_module(module_name, *info)


def init_model(model, param=None):
    """Save initial params or load params to resume."""
    if param is None:
        return False
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
        return True


def init_optimizer(optimizer, state=None):
    """Save initial state or load state to resume."""
    return init_model(optimizer, state)


def l2_norm(var):
    """Calculate L2 norm of each sample."""
    if var.ndim > 1:
        if np.asarray(var.shape[1:]).prod() > 1:
            return F.sqrt(F.sum(var * var,
                          axis=tuple(range(1, var.ndim))))
        else:
            var = F.reshape(var, (-1,))

    return abs(var)
