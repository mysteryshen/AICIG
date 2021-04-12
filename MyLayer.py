from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

class MyLayer1(Layer):
    def __init__(self, batch_size=100,output_dim=100, class_num=10,sigma=5, **kwargs):
        self.batch_size=batch_size
        self.output_dim = output_dim
        self.class_num = class_num
        self.sigma_value = float(sigma)
        super(MyLayer1, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self._mu = self.add_weight(name='mu',
                                    shape=((self.class_num,input_shape[1])),
                                    initializer=tf.random_uniform_initializer(-1,1),
                                    trainable=True)

        self._sigma = self.add_weight(name='sigma',
                                    shape=((self.class_num,input_shape[1])),
                                    initializer=tf.constant_initializer(self.sigma_value),
                                    trainable=True)

        super(MyLayer1, self).build(input_shape)  # Be sure to call this at the end

    def call(self, z):
        repeat_count = self.batch_size//self.class_num
        mu = K.repeat_elements(self._mu,repeat_count,axis=0)
        sigma = K.repeat_elements(self._sigma,repeat_count,axis=0)
        return mu + z * sigma

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.output_dim)

def create_custom_objects():
    instance_holder = {"instance": None}
    class ClassWrapper(MyLayer1):
        def __init__(self, *args, **kwargs):
            instance_holder["instance"] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)
    return {"ClassWrapper": ClassWrapper ,"MyLayer1": ClassWrapper}

def R_loss(mu, sigma,d_number,beta_value):
    class_num = mu.shape[0]

    #calculate d_mean
    d_sum = tf.constant(0.0)
    for i in range(class_num - 1):
        for j in range(i + 1, class_num):
            distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(mu[i], mu[j]))))
            d_sum = tf.add(d_sum, distance)
    d_mean = tf.div(d_sum, d_number)

    # calculate sigma_mean
    sigma_mean = tf.reduce_mean(sigma)

    #calculate R
    R = tf.div(d_mean, sigma_mean)
    loss_R = tf.multiply(beta_value,R)
    return loss_R
