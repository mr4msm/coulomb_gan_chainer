# -*- coding: utf-8 -*-
"""Generate image using a Generator trained with GAN."""

import argparse
import chainer
import cv2
import numpy as np
import os
import random
from chainer import cuda

from commons import initialize_model, load_module


def parse_arguments():
    """Define and parse positional/optional arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config',
        help='a configuration file in which Generator is defined (.py)'
    )
    parser.add_argument(
        '-g', '--gpu', default=-1, type=int,
        help='GPU ID (negative value indicates CPU)'
    )
    parser.add_argument(
        '-n', '--n_lines', default=8, type=int,
        help='generate a n_lines * n_lines concatenated image'
    )
    parser.add_argument(
        '-o', '--output', default='./generated_image.png',
        help='a path of output image file'
    )
    parser.add_argument(
        '-p', '--param', default=None,
        help='trained parameters saved as serialized array (.npz | .h5)'
    )
    parser.add_argument(
        '-r', '--random_seed', default=0, type=int,
        help='random seed to determine z vector in latent space'
    )

    return parser.parse_args()


def generate_image(config, gpu_id=-1, n_lines=8, param=None, random_seed=0):
    """Generate image using a Generator trained with GAN."""
    # setup network model and constant values to control image generation
    z_vec_dim = config.Z_VECTOR_DIM
    height = config.HEIGHT
    width = config.WIDTH
    channel = config.CHANNEL
    image_generator = config.Generator()

    # load parameters for Generator
    initialize_model(image_generator, param)

    # set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    # determine z vector in latent space
    if n_lines < 1:
        n_lines = 1
    z = np.random.normal(loc=0., scale=1.,
                         size=(n_lines ** 2, z_vec_dim)).astype('float32')

    # set current device and copy model to it
    if gpu_id >= 0:
        cuda.get_device(gpu_id).use()
        image_generator.to_gpu(device=gpu_id)
        z = cuda.to_gpu(z)

    # set global configuration
    chainer.global_config.enable_backprop = False
    chainer.global_config.train = False

    # generate image
    gen_out = ((cuda.to_cpu(image_generator(z).data) + 1.) *
               255.99 / 2.).astype(np.uint8).transpose((0, 2, 3, 1))

    image = np.empty((n_lines * height, n_lines * width, channel),
                     dtype=np.uint8)
    for y in range(n_lines):
        for x in range(n_lines):
            image[y * height:(y + 1) * height,
                  x * width:(x + 1) * width] = gen_out[y * n_lines + x]

    return image


if __name__ == '__main__':
    # parse arguments
    args = parse_arguments()
    config = load_module(args.config)

    # make output directory, if needed
    out_dir, _ = os.path.split(args.output)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print('mkdir ' + out_dir)
    assert os.path.isdir(out_dir)

    cv2.imwrite(args.output,
                generate_image(config, gpu_id=args.gpu,
                               n_lines=args.n_lines,
                               param=args.param,
                               random_seed=args.random_seed))
    print('save ' + args.output)
