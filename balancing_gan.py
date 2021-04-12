"""
(C) Copyright IBM Corporation 2018

All rights reserved. This program and the accompanying materials
are made available under the terms of the Eclipse Public License v1.0
which accompanies this distribution, and is available at
http://www.eclipse.org/legal/epl-v10.html
"""

from collections import defaultdict
from time import time
import keras.backend as K
from result_logger import ResultLogger
K.set_image_dim_ordering('th')

from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import os
import re
import numpy as np

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization,Activation
from utils import save_image_array
from MyLayer import *

class BalancingGAN:
    def __init__(self, classes, target_class_id, tag,
                 # Set dratio_mode, and gratio_mode to 'rebalance' to bias the sampling toward the minority class
                 # No relevant difference noted
                 dratio_mode="uniform", gratio_mode="uniform",
                 adam_lr=0.00005, latent_size=100,
                 res_dir="./res-tmp", image_shape=[3, 32, 32], min_latent_res=8,
                 N=10,batch_size=100,sigma=5,beta=5):
        self.beta = beta
        self.gratio_mode = gratio_mode
        self.dratio_mode = dratio_mode
        self.classes = classes
        self.target_class_id = target_class_id
        self.nclasses = len(classes)
        self.latent_size = latent_size
        self.res_dir = res_dir
        self.channels = image_shape[0]
        self.resolution = image_shape[1]
        if self.resolution != image_shape[2]:
            print("Error: only squared images currently supported by balancingGAN")
            exit(1)

        # self.min_latent_res = min_latent_res

        # Initialize learning variables
        self.adam_lr = adam_lr
        self.adam_beta_1 = 0.5

        # Initialize stats
        self.train_history = defaultdict(list)
        self.test_history = defaultdict(list)
        self.trained = False

        # Build final_latent
        self.build_latent_use_Gaussian(latent_size=latent_size,N=self.nclasses,batch_size=batch_size,sigma=sigma)
        self.final_latent.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            loss='sparse_categorical_crossentropy'
        )

        #Build generator
        self.build_generator(latent_size,N, init_resolution=min_latent_res)
        self.generator.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            loss='sparse_categorical_crossentropy'
        )

        # Build classifier
        self.build_classifier(min_latent_res=min_latent_res)
        self.classifier.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            loss='sparse_categorical_crossentropy'
        )
        # Define combined for training generator with final_latent.
        latent_gen = Input(batch_shape=(batch_size,latent_size))
        # class_label = Input(batch_shape=(batch_size,N))
        z_withclass = self.final_latent(latent_gen)
        fake = self.generator(z_withclass)
        aux = self.classifier(fake)

        self.final_latent.trainable = True
        self.classifier.trainable = False
        self.generator.trainable = True

        self.combined = Model(inputs=latent_gen, outputs=aux)
        # loss of regularization
        weights = self.combined.get_layer(index=1).weights
        mu_t = weights[0]
        sigma_t = weights[1]
        number=(self.nclasses*(self.nclasses-1)/2.0)
        # the value that R tends to
        beta_value = tf.constant(float(beta))
        loss_R = R_loss(mu_t,sigma_t,number,beta_value)
        self.combined.add_loss(loss_R)
        self.combined.compile(
            optimizer=Adam(lr=self.adam_lr, beta_1=self.adam_beta_1),
            loss='sparse_categorical_crossentropy'
        )
        self.tag = tag
        self.result_logger = ResultLogger(tag, self.res_dir, verbose=True)

    def generate_latent(self, num, latent_size=100, mode_z='uniform'):
        if mode_z == 'uniform':
            gen_z = np.random.uniform(-1.0, 1.0, [num, latent_size]).astype(np.float32)
        else:
            gen_z = np.random.normal(0.0, 1.0, [num,latent_size]).astype(np.float32)
        return gen_z

    def generate_image_labels(self,class_num=10,sample_num=[]):
        generated_images_labels = []
        for i in range(class_num):
            for j in range(sample_num[i]):
                generated_images_labels.append(i)
        return generated_images_labels

    def build_latent_use_Gaussian(self,latent_size=100,N=10,batch_size=100,sigma=5):
        latent = Input(batch_shape=(batch_size,latent_size))
        reparamter = MyLayer1(batch_size=batch_size,output_dim=latent_size, class_num=N,sigma=sigma)
        reparamter_res = reparamter(latent)
        self.final_latent = Model(inputs =latent, outputs = reparamter_res)

    def build_generator(self, latent_size=100, N=10, init_resolution=8):
        resolution = self.resolution
        channels = self.channels

        cnn = Sequential()
        cnn.add(Dense(1024, input_dim=latent_size, use_bias=False))
        cnn.add(BatchNormalization(momentum=0.9))
        cnn.add(Activation('relu'))

        cnn.add(Dense(128 * init_resolution * init_resolution, use_bias=False))
        cnn.add(BatchNormalization(momentum=0.9))
        cnn.add(Activation('relu'))

        cnn.add(Reshape((128, init_resolution, init_resolution)))
        crt_res = init_resolution

        # upsample
        while crt_res != resolution:
            cnn.add(UpSampling2D(size=(2, 2)))
            if crt_res < resolution / 2:
                cnn.add(Conv2D(
                    256, (5, 5), padding='same',
                    kernel_initializer='glorot_normal', use_bias=False)
                )
                cnn.add(BatchNormalization(momentum=0.9))
                cnn.add(Activation('relu'))
            else:
                cnn.add(Conv2D(128, (5, 5), padding='same',
                               kernel_initializer='glorot_normal', use_bias=False))
                cnn.add(BatchNormalization(momentum=0.9))
                cnn.add(Activation('relu'))

            crt_res = crt_res * 2
            assert crt_res <= resolution, \
                "Error: final resolution [{}] must equal i*2^n. Initial resolution i is [{}]. n must be a natural number.".format(
                    resolution, init_resolution)
        # FIXME: sigmoid here
        cnn.add(Conv2D(channels, (2, 2), padding='same',
                       activation='tanh', kernel_initializer='glorot_normal', use_bias=False))
        # This is the latent z space
        latent = Input(shape=(latent_size,))

        fake_image_from_latent = cnn(latent)

        # The input-output interface
        self.generator = Model(inputs=latent, outputs=fake_image_from_latent)

    def _build_common_encoder(self, image, min_latent_res=8):
        resolution = self.resolution
        channels = self.channels

        # build a relatively standard conv net, with LeakyReLUs as suggested in ACGAN
        cnn = Sequential()

        cnn.add(Conv2D(32, (3, 3), padding='same', strides=(2, 2),
                       input_shape=(channels, resolution, resolution), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(128, (3, 3), padding='same', strides=(2, 2), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        cnn.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), use_bias=True))
        cnn.add(LeakyReLU())
        cnn.add(Dropout(0.3))

        while cnn.output_shape[-1] > min_latent_res:
            cnn.add(Conv2D(256, (3, 3), padding='same', strides=(2, 2), use_bias=True))
            cnn.add(LeakyReLU())
            cnn.add(Dropout(0.3))

            cnn.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1), use_bias=True))
            cnn.add(LeakyReLU())
            cnn.add(Dropout(0.3))

        cnn.add(Flatten())

        features = cnn(image)
        return features

    def build_classifier(self, min_latent_res=8):
        resolution = self.resolution
        channels = self.channels
        image = Input(shape=(channels, resolution, resolution))
        features = self._build_common_encoder(image, min_latent_res)
        aux = Dense(
            self.nclasses * 2, activation='softmax', name='auxiliary'
        )(features)
        aux1 = Dense(
            self.nclasses, activation='softmax', name='classifier'
        )(aux)
        self.classifier = Model(inputs=image, outputs=[aux,aux1])

    def get_batch_count(self,class_num,batch_size,gen_class_ration):
        sample = np.random.choice([i for i in range(class_num)], size=batch_size, replace=True, p=gen_class_ration)
        batch_num = [0]*class_num
        for x in sample:
            batch_num[x]+=1
        batch_num= np.array(batch_num)
        return batch_num

    def _train_one_epoch(self, bg_train,class_num,batch_size=100, latent_size=100,mode_z='uniform',gen_class_ration=[]):
        epoch_classifier_loss = []
        epoch_gen_loss = []

        for image_batch, label_batch in bg_train.next_batch():
            ################## Train Classifier ##################
            X = image_batch
            aux_y = label_batch
            noise_gen = self.generate_latent(batch_size, latent_size, mode_z)
            gen_counts = self.get_batch_count(class_num, batch_size, gen_class_ration)
            weights = self.final_latent.get_layer(index=1).get_weights()
            mu = weights[0]
            sigma=weights[1]
            final_mu = np.repeat(mu,gen_counts,axis=0)
            final_sigma = np.repeat(sigma,gen_counts,axis=0)
            final_z = final_mu+final_sigma*noise_gen
            fake_label = self.generate_image_labels(class_num, gen_counts)
            generated_images = self.generator.predict(final_z, verbose=0,batch_size=batch_size)
            X = np.concatenate((X,generated_images),axis=0)
            class_np = np.array([self.nclasses]*batch_size)
            fake_label_np = np.array(fake_label)
            fake_label_train = class_np+fake_label_np

            aux_y = np.concatenate((aux_y,fake_label_train),axis = 0)
            aux1_y = np.concatenate((label_batch,fake_label_np),axis = 0)
            epoch_classifier_loss.append(self.classifier.train_on_batch(X, [aux_y,aux1_y]))

            ################## Train Generator ##################
            noise_gen = self.generate_latent(batch_size, latent_size, mode_z)
            gen_label = self.generate_image_labels(class_num,[batch_size//class_num]*class_num)
            loss_gen = self.combined.train_on_batch(noise_gen, [gen_label,gen_label])
            # loss_R = loss_gen[0] - loss_gen[1] - loss_gen[2]
            epoch_gen_loss.append(loss_gen)

        # return statistics: generator loss,
        return (
            np.mean(np.array(epoch_classifier_loss), axis=0),
            np.mean(np.array(epoch_gen_loss), axis=0)
        )

    def _get_lst_bck_name(self, element):
        # Find last bck name
        files = [
            f for f in os.listdir(self.res_dir)
            if re.match(r'bck_c_{}'.format(self.target_class_id) + "_" + element, f)
        ]
        if len(files) > 0:
            fname = files[0]
            e_str = os.path.splitext(fname)[0].split("_")[-1]
            epoch = int(e_str)
            return epoch, fname
        else:
            return 0, None


    def backup_point(self, epoch,epochs=100):
        if epoch%50 == 0 or epoch ==(epochs-1) or epoch ==(epochs-2) or epoch ==(epochs-3):
            classifier_fname = "{}/bck_c_{}_classifier_e_{}.h5".format(self.res_dir, self.target_class_id, epoch)
            self.classifier.save(classifier_fname)
            generator_fname = "{}/bck_c_{}_generator_e_{}.h5".format(self.res_dir, self.target_class_id, epoch)
            self.generator.save(generator_fname)
        final_latent_fname = "{}/bck_c_{}_latent_e_{}.h5".format(self.res_dir, self.target_class_id, epoch)
        self.final_latent.save(final_latent_fname)


    def train(self, bg_train, bg_test, epochs=100,class_num=10,latent_size=100,mode_z ='uniform',batch_size=100,gen_class_ration=[]):
        if not self.trained:
            # Class actual ratio
            self.class_aratio = bg_train.get_class_probability()
            fixed_latent = self.generate_latent(batch_size, latent_size, mode_z)

            # Train
            start_e=0
            for e in range(start_e, epochs):
                start_time = time()
                # Train
                print('GAN train epoch: {}/{}'.format(e, epochs))
                train_classifier_loss, train_gen_loss = self._train_one_epoch(bg_train,class_num,batch_size = batch_size, mode_z=mode_z,gen_class_ration=gen_class_ration)

                loss_R = train_gen_loss[0] - train_gen_loss[1] - train_gen_loss[2]
                self.result_logger.add_training_metrics1(float(train_gen_loss[0]), float(train_gen_loss[1]),
                                                        float(train_gen_loss[2]),float(loss_R),
                                                        float(train_classifier_loss[0]), float(train_classifier_loss[1]),
                                                        float(train_classifier_loss[2]),
                                                        time() - start_time)
                # Test #
                test_loss = self.classifier.evaluate(bg_test.dataset_x, [bg_test.dataset_y, bg_test.dataset_y],
                                                     verbose=False)
                self.result_logger.add_testing_metrics(test_loss[0], test_loss[1], test_loss[2])
                probs_0,probs_1 = self.classifier.predict(bg_test.dataset_x, batch_size=batch_size, verbose=True)
                final_probs = probs_1
                predicts = np.argmax(final_probs, axis=-1)
                self.result_logger.save_prediction(e, bg_test.dataset_y, predicts, probs_0,probs_1,epochs=epochs)
                self.result_logger.save_metrics()
                print(
                    "train_classifier_loss {},\ttrain_gen_loss {},\t".format(
                        train_classifier_loss, train_gen_loss
                    ))

                # Save sample images
                if e % 1 == 0:
                    final_latent = self.final_latent.predict(fixed_latent, batch_size=batch_size)
                    generated_images = self.generator.predict(final_latent, verbose=0, batch_size=batch_size)
                    img_samples = generated_images / 2. + 0.5  # 从[-1,1]恢复到[0,1]之间的值
                    save_image_array(
                        img_samples,
                        '{}/plot_epoch_{}.png'.format(self.res_dir, e),
                        batch_size=batch_size,
                        class_num=10
                    )
                if e % 1 == 0:
                    self.backup_point(e,epochs)
            self.trained = True