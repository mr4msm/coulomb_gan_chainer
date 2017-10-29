# -*- coding: utf-8 -*-
"""Train Discriminator and Generator using Coulomb GAN."""

import argparse
import chainer
import numpy as np
import os
from chainer import cuda, optimizer
from chainer import computational_graph as graph
from chainer import functions as F
from datetime import datetime as dt

from batch_generator import ImageBatchGenerator
from commons import ModelOptimizerSet
from commons import load_module
from commons import init_model, init_optimizer


def parse_arguments():
    """Define and parse positional/optional arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config',
        help='a configuration file in which networks, optimizers and hyper params are defined (.py)'
    )
    parser.add_argument(
        'dataset',
        help='a text file in which image files are listed'
    )
    parser.add_argument(
        '-c', '--computational_graph', action='store_true',
        help='if specified, build computational graph'
    )
    parser.add_argument(
        '-g', '--gpu', default=-1, type=int,
        help='GPU ID (negative value indicates CPU)'
    )
    parser.add_argument(
        '-o', '--output', default='./',
        help='a directory in which output files will be stored'
    )
    parser.add_argument(
        '-p', '--param_dis', default=None,
        help='trained parameters for Discriminator saved as serialized array (.npz | .h5)'
    )
    parser.add_argument(
        '-P', '--param_gen', default=None,
        help='trained parameters for Generator saved as serialized array (.npz | .h5)'
    )
    parser.add_argument(
        '-s', '--state_dis', default=None,
        help='optimizer state for Discriminator saved as serialized array (.npz | .h5)'
    )
    parser.add_argument(
        '-S', '--state_gen', default=None,
        help='optimizer state for Generator saved as serialized array (.npz | .h5)'
    )

    return parser.parse_args()


def plummer_kernel(a, b, dim=1, eps=1, xp=np):
    squared_l2 = (a - b) ** 2
    if squared_l2.ndim > 1:
        squared_l2 = xp.sum(squared_l2, axis=tuple(range(1, squared_l2.ndim)))
    inverse = xp.sqrt((squared_l2 + eps ** 2) ** dim)
    return 1. / inverse


def potential(a, y_batch, x_batch, dim=1, eps=1, xp=np):
    y_size = len(y_batch)
    y_sum = 0.
    for idx in range(y_size):
        y_sum += plummer_kernel(a, y_batch[idx: idx + 1],
                                dim=dim, eps=eps, xp=xp)

    x_size = len(x_batch)
    x_sum = 0.
    for idx in range(x_size):
        x_sum += plummer_kernel(a, x_batch[idx: idx + 1],
                                dim=dim, eps=eps, xp=xp)

    return x_sum / x_size - y_sum / y_size


if __name__ == '__main__':
    # parse arguments
    args = parse_arguments()
    config = load_module(args.config)
    out_dir = args.output
    gpu_id = args.gpu

    # make output directory, if needed
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print('mkdir ' + out_dir)
    assert os.path.isdir(out_dir)

    # setup network model, optimizer, and constant values to control training
    z_vec_dim = config.Z_VECTOR_DIM
    batch_size = config.BATCH_SIZE
    update_max = config.UPDATE_MAX
    update_save_params = config.UPDATE_SAVE_PARAMS
    kernel_dim = getattr(config, 'KERNEL_DIM', 1)
    kernel_eps = getattr(config, 'KERNEL_EPS', 1)

    model_dis = config.Discriminator()
    optimizer_dis = config.OPTIMIZER_DIS
    optimizer_dis.setup(model_dis)
    decay_d = getattr(config, 'DECAY_RATE_DIS', 1e-7)
    optimizer_dis.add_hook(optimizer.WeightDecay(decay_d))
    model_opt_set_dis = ModelOptimizerSet(model_dis, optimizer_dis)

    model_gen = config.Generator()
    optimizer_gen = config.OPTIMIZER_GEN
    optimizer_gen.setup(model_gen)
    decay_g = getattr(config, 'DECAY_RATE_GEN', 1e-7)
    optimizer_gen.add_hook(optimizer.WeightDecay(decay_g))
    model_opt_set_gen = ModelOptimizerSet(model_gen, optimizer_gen)

    # setup batch generator
    with open(args.dataset, 'r') as f:
        input_files = [line.strip() for line in f.readlines()]
    batch_generator = ImageBatchGenerator(input_files, batch_size,
                                          config.HEIGHT, config.WIDTH,
                                          channel=config.CHANNEL,
                                          shuffle=True,
                                          flip_h=getattr(config, 'FLIP_H', False))
    sample_num = batch_generator.n_samples

    # show some settings
    print('sample num = {}'.format(sample_num))
    print('mini-batch size = {}'.format(batch_size))
    print('max update count = {}'.format(update_max))
    print('updates per saving params = {}'.format(update_save_params))
    print('plummer kernel dimension = {}'.format(kernel_dim))
    print('plummer kernel epsilon = {}'.format(kernel_eps))

    # save or load initial parameters for Discriminator
    if not init_model(model_dis, param=args.param_dis):
        model_opt_set_dis.save_model('dis', out_dir=out_dir)
    # save or load initial optimizer state for Discriminator
    if not init_optimizer(optimizer_dis, state=args.state_dis):
        model_opt_set_dis.save_optimizer('dis', out_dir=out_dir)
    # save or load initial parameters for Generator
    if not init_model(model_gen, param=args.param_gen):
        model_opt_set_gen.save_model('gen', out_dir=out_dir)
    # save or load initial optimizer state for Generator
    if not init_optimizer(optimizer_gen, state=args.state_gen):
        model_opt_set_gen.save_optimizer('gen', out_dir=out_dir)

    # set current device and copy model to it
    xp = np
    if gpu_id >= 0:
        cuda.get_device(gpu_id).use()
        model_gen.to_gpu(device=gpu_id)
        model_dis.to_gpu(device=gpu_id)
        xp = cuda.cupy

    # set global configuration
    chainer.global_config.enable_backprop = True
    chainer.global_config.train = True

    print('** chainer global configuration **')
    chainer.global_config.show()

    initial_t = optimizer_gen.t
    sum_count = 0
    sum_loss_dis = 0.
    sum_loss_gen = 0.

    # training loop
    while optimizer_gen.t < update_max:
        y = next(batch_generator)
        if gpu_id >= 0:
            y = cuda.to_gpu(y, device=gpu_id)

        z = xp.random.normal(loc=0., scale=1.,
                             size=(batch_size, z_vec_dim)).astype('float32')
        x = model_gen(z)

        loss_dis = F.sum(F.square(model_dis(y) - potential(y, y, x.data,
                                                           dim=kernel_dim,
                                                           eps=kernel_eps,
                                                           xp=xp)))
        loss_dis += F.sum(F.square(model_dis(x.data) - potential(x.data, y, x.data,
                                                                 dim=kernel_dim,
                                                                 eps=kernel_eps,
                                                                 xp=xp)))
        loss_dis /= batch_size

        # update Discriminator
        model_dis.cleargrads()
        loss_dis.backward()
        optimizer_dis.update()

        sum_loss_dis += float(loss_dis.data)

        loss_gen = F.sum(model_dis(x))
        loss_gen /= batch_size

        # update Generator
        model_gen.cleargrads()
        loss_gen.backward()
        optimizer_gen.update()

        sum_loss_gen += float(loss_gen.data)
        sum_count += 1

        # show losses
        print('{0}: update # {1:09d}: D loss = {2:6.4e}, G loss = {3:6.4e}'.format(
            str(dt.now()), optimizer_gen.t,
            float(loss_dis.data), float(loss_gen.data)))

        # output computational graph, if needed
        if args.computational_graph and optimizer_gen.t == (initial_t + 1):
            with open('graph.dot', 'w') as o:
                o.write(graph.build_computational_graph((loss_dis, loss_gen)).dump())
            print('graph generated')

        # show mean losses, save interim trained parameters and optimizer states
        if optimizer_gen.t % update_save_params == 0:
            print('{0}: mean of latest {1:06d} in {2:09d} updates : D loss = {3:7.5e}, G loss = {4:7.5e}'.format(
                str(dt.now()), sum_count, optimizer_gen.t, sum_loss_dis / sum_count, sum_loss_gen / sum_count))
            sum_count = 0
            sum_loss_dis = 0.
            sum_loss_gen = 0.

            model_opt_set_gen.save('gen', out_dir=out_dir)
            model_opt_set_dis.save('dis', out_dir=out_dir)

    # save final trained parameters and optimizer states
    model_opt_set_gen.save('gen', out_dir=out_dir)
    model_opt_set_dis.save('dis', out_dir=out_dir)
