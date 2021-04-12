"""
(C) Copyright IBM Corporation 2018
All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

import numpy as np
from optparse import OptionParser
import balancing_gan as bagan
from batch_generator import BatchGenerator as BatchGenerator
import os
import tensorflow as tf
import keras.backend as K
import keras.backend.tensorflow_backend as ktf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["OMP_NUM_THREADS"] = "8"

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=config)

def main(seed, dataset,sigma,beta):
    # Collect arguments
    argParser = OptionParser()

    argParser.add_option("-u", "--unbalance", default=0.05,
                         action="store", type="float", dest="unbalance",
                         help="Unbalance factor u. The minority class has at most u * otherClassSamples instances.")

    argParser.add_option("-s", "--random_seed", default=seed,
                         action="store", type="int", dest="seed",
                         help="Random seed for repeatable subsampling.")

    argParser.add_option("-d", "--sampling_mode_for_discriminator", default="uniform",
                         action="store", type="string", dest="dratio_mode",
                         help="Dratio sampling mode (\"uniform\",\"rebalance\").")

    argParser.add_option("-g", "--sampling_mode_for_generator", default="uniform",
                         action="store", type="string", dest="gratio_mode",
                         help="Gratio sampling mode (\"uniform\",\"rebalance\").")

    argParser.add_option("-e", "--epochs", default=150,
                         action="store", type="int", dest="epochs",
                         help="Training epochs.")

    argParser.add_option("-l", "--learning_rate", default=0.0001,
                         action="store", type="float", dest="adam_lr",
                         help="Training learning rate.")

    argParser.add_option("-c", "--target_class", default=-1,
                         action="store", type="int", dest="target_class",
                         help="If greater or equal to 0, model trained only for the specified class.")

    argParser.add_option("-D", "--dataset", default=dataset,
                         action="store", type="string", dest="dataset",
                         help="Either 'MNIST', or 'CIFAR10'.")
    argParser.add_option("-p", "--pretrain", default=False, action="store", dest="pretrain")
    argParser.add_option("-m", "--mode_of_z", default='uniform', action="store",type='string',dest="mode_z",help='uniform/gaussian')

    (options, args) = argParser.parse_args()

    assert (options.unbalance <= 1.0 and options.unbalance > 0.0), "Data unbalance factor must be > 0 and <= 1"

    print("Executing My Model.")

    # Read command line parameters
    np.random.seed(options.seed)
    unbalance = options.unbalance
    gratio_mode = options.gratio_mode
    dratio_mode = options.dratio_mode
    gan_epochs = options.epochs
    adam_lr = options.adam_lr
    mode_z = options.mode_z
    max_count_of_class = 16000
    latent_size = 100
    class_num = 5
    per_class_num = 20
    batch_size = class_num * per_class_num
    dataset_name = options.dataset
    # tag = '%s_seed%01d' % (dataset_name, options.seed)
    tag = '%s_seed%01d_sigma%.1f_beta%.2f' % (dataset_name, options.seed,sigma,beta)
    # Set channels for mnist.
    channels = 1 if dataset_name in ('mnist', 'fashion') else 3
    print('Using dataset: ', dataset_name)

    # Result directory
    res_dir = './res/%s' % (tag)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # Read initial data.
    print("read input data...")

    bg_train_full = BatchGenerator(BatchGenerator.TRAIN, options.seed, batch_size, dataset=dataset_name)
    bg_test = BatchGenerator(BatchGenerator.TEST, options.seed, batch_size, dataset=dataset_name)
    # class_num = bg_train_full.get_num_classes()
    print("input data loaded...")

    shape = bg_train_full.get_image_shape()

    min_latent_res = shape[-1]
    while min_latent_res > 8:
        min_latent_res = min_latent_res / 2
    min_latent_res = int(min_latent_res)
    classes = bg_train_full.get_label_table()

    # For all possible minority classes.
    target_classes = np.array(range(len(classes)))

    # Unbalance the training set.
    bg_train_partial = bg_train_full
    print('Class counters: ', bg_train_partial.per_class_count)

    gen_class_count = [(max_count_of_class - i) for i in bg_train_partial.per_class_count]
    gen_class_count = np.array(gen_class_count)
    gen_class_ration = gen_class_count / sum(gen_class_count)

    gan = bagan.BalancingGAN(
        target_classes, 0, tag, dratio_mode=dratio_mode, gratio_mode=gratio_mode,
        adam_lr=adam_lr, res_dir=res_dir, image_shape=shape, min_latent_res=min_latent_res,N=class_num,batch_size=batch_size,sigma=sigma,beta=beta
    )

    gan.train(bg_train_partial, bg_test, epochs=gan_epochs,class_num=class_num,latent_size=latent_size,mode_z=mode_z,batch_size=batch_size,gen_class_ration=gen_class_ration)

def get_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

if __name__ == '__main__':
    dataset = 'celeba'
    sigmas = [10]
    betas = [0.75]
    for sigma in sigmas:
        for beta in betas:
            for seed in range(1, 2):
                K.clear_session()
                ktf.set_session(get_session())
                main(seed, dataset, sigma,beta)
                print("AICIG %s seed %d finished" % (dataset, seed))
