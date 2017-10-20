This is an implementation of [Coulomb GANs: Provably Optimal Nash Equilibria via Potential Fields](https://arxiv.org/abs/1708.08819) in [Chainer](https://github.com/chainer/chainer) >= v2.

# Requirements
Chainer >= v2, OpenCV, etc.  
The scripts work on Python 2.7.13 and 3.6.1.

# How to generate images
```
$ python generate_image.py example_food-101/config.py -p example_food-101/trained-params_gen_update-000128000.npz
```
You can generate various images by changing the random_seed option.
```
$ python generate_image.py example_food-101/config.py -r 1 -p example_food-101/trained-params_gen_update-000128000.npz
```

## Example Food-101
![example_image_food-101](https://raw.githubusercontent.com/mr4msm/coulomb_gan_chainer/master/example_food-101/example_update-000128000.png)

# Dataset
I resized the images to 64x64 before training.
* [Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)  
Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc. Food-101 -- Mining Discriminative Components with Random Forests. European Conference on Computer Vision, 2014.
