import random
import numpy as np
import tensorflow as tf


# a concise way to implement C-like structures. 
# ref: https://stackoverflow.com/questions/1878710/struct-objects-in-python
config = lambda: 0
# note: put a comma after `tf.random_normal` will make it viewed as a tuple cuz it's not dict!...
config.weights_initializer = tf.random_normal
# config.weights_initializer = tf.diag
config.bias_initializer = tf.zeros
config.feature_dimension = 784
config.num_of_class = 10
config.training_epochs = 250


class NN():
    def __init__(self):
        self.config = config
        self.W = tf.cast(tf.Variable(self.config.weights_initializer([
            self.config.feature_dimension, 
            self.config.num_of_class
        ])), dtype=tf.float32)
#         self.W = tf.cast(tf.Variable(self.config.weights_initializer(
#             [1, 1], 
#         )), dtype=tf.float32)
#         self.W = tf.cast(tf.Variable(initial_value = np.identity(2)), dtype=tf.float32)
        
        self.b = tf.Variable(self.config.bias_initializer(shape= \
                            [self.config.num_of_class, ]))
#         print(tf.shape(self.W))
        
        config_ = tf.ConfigProto()
        config_.gpu_options.allow_growth = True
    
        self.sess = tf.Session(config=config_)
        self.sess.run(tf.global_variables_initializer())
        
        self.W_holder = tf.placeholder(tf.float32, [
            self.config.feature_dimension,
            self.config.num_of_class
        ])
        self.b_holder = tf.placeholder(tf.float32,
                                       [self.config.num_of_class, ])
        
        self.x_s_holder = tf.placeholder(tf.float32, 
                                         [None,self.config.feature_dimension])
        self.y_s_holder = tf.placeholder(tf.float32, 
                                         [None, self.config.num_of_class])
        self.logits_holder = self.logits_holder()
#         self.logits_holder = tf.nn.softmax(self.logits_holder)

#         indices = tf.transpose(tf.convert_to_tensor([
#             list(range(100)),
#             tf.argmax(self.y_s_holder, axis=1)
#         ]))
#         self.loss_holder = tf.reduce_mean(tf.negative(tf.log(tf.gather_nd(self.logits_holder, indices))))
#         self.loss_holder = tf.reduce_mean(
#             tf.square(self.logits_holder - self.y_s_holder))
        self.loss_holder = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_holder,
                                                    labels=self.y_s_holder)
        )
#         self.loss_holder = tf.reduce_mean(
#             -tf.reduce_sum(self.y_s_holder * tf.log(self.logits_holder), reduction_indices=[1])
#         )

        self.acc_holder = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(self.logits_holder, axis=1), 
            tf.argmax(self.y_s_holder, axis=1)
        ), tf.float32))
        self.train_op_holder = tf.train.GradientDescentOptimizer(0.5). \
        minimize(self.loss_holder)
        
        
    def __del__(self):
        if hasattr(self, 'sess'):
            self.sess.close()
        
    
    def fit(self, x_s_train, y_s_train, steps=None,
            params_trace_recording=False):
        if not steps:
            steps = self.config.training_epochs
        W_trace = [self.sess.run(self.W)]
        b_trace = [self.sess.run(self.b)]
        
#         import pdb
#         pdb.set_trace()
        
        for step in range(steps):
            indices_batch = random.sample(
                range(len(x_s_train)), 1000)
            _ = self.sess.run(
                [self.train_op_holder], 
                feed_dict={
                    self.x_s_holder: x_s_train[indices_batch],
                    self.y_s_holder: y_s_train[indices_batch]
                }
            )
            
            
            if step % 5 == 0:
                loss, acc = self.sess.run(
                    [self.loss_holder, self.acc_holder], \
                    feed_dict={
                        self.x_s_holder: x_s_train,
                        self.y_s_holder: y_s_train
                    }
                )
                print(loss, '|', acc)
        
            if params_trace_recording or step == self.config.training_epochs - 1:
                W_trace.append(self.sess.run(self.W))
                b_trace.append(self.sess.run(self.b))
#             if step % 1000 == 0: print(loss, acc)
        
        W_trace = np.array(W_trace)
        b_trace = np.array(b_trace)
        print('training completed.')
        return {'W': W_trace, 'b': b_trace, 'W_grad': 0, 'b_grad': 0, 'loss':0}
    
    
    def logits_holder(self, custom_params=None):
        if custom_params:
            logits_holder = tf.matmul(self.x_s_holder, self.W_holder) + \
            self.b_holder
        else:
            logits_holder = tf.matmul(self.x_s_holder, self.W) + self.b
        return logits_holder


    def logits(self, x_s, W=None, b=None):
        assert (W is None) == (b is None)
        feed_dict = {self.x_s_holder: x_s}
        if W is not None and b is not None:
            feed_dict[self.W_holder] = W
            feed_dict[self.b_holder] = b
            logits_holder = tf.matmul(self.x_s_holder, self.W_holder) + \
            self.b_holder
            logits_holder = tf.nn.softmax(logits_holder)
            return self.sess.run(logits_holder, feed_dict=feed_dict)
    
    
    
    # predict is the actual prediction result based the logits.
#     def predict(self, x_s, custom_params=None):
#         predict = tf.argmax(self.logits(custom_params), axis=1)
#         feed_dict = {self.x_s_holder: x_s}
#         if custom_params:
#             feed_dict[self.W_holder] = custom_params['W']
#             feed_dict[self.b_holder] = custom_params['b']
#         predict = self.sess.run(predict, feed_dict=feed_dict)
#         return predict

    def predict(self, x_s, W=None, b=None, one_hot=True):
        x_s = np.array(x_s)
        assert (W is None) == (b is None), 'W:%s |b:%s' % (W, b)
        max_indices = np.argmax(self.logits(x_s, W, b), 1)
        if not one_hot:
            predict = max_indices
        else:
            predict = np.zeros_like(x_s).astype(np.int)
            predict[range(len(x_s)), max_indices] = 1
        return predict
    
                 
    def accuracy(self, x_s, y_s, W=None, b=None):
        x_s, y_s = np.array(x_s), np.array(y_s)
        assert x_s.shape == y_s.shape
        custom_params = {'W': W, 'b':b}
        predict = self.predict(x_s, W, b, one_hot=False)
        return np.mean(np.equal(predict, np.argmax(y_s, 1)))

    
if __name__ == '__main__':
    # data preparation.
    x = np.linspace(-1, 1, 50)
    y = 0.5 * np.sin(3.1*(x-0.5))
    x_s_trian = np.hstack([(x,y+0.4),(x,y-0.45)])
    y_s_train = np.array([[1, 0]]*50 + [[0, 1]]*50)

    # training process of NN.
    nn = NN(config)
    W_trace, b_trace = nn.fit(x_s_train, y_s_train,
                              params_trace_recording=True)