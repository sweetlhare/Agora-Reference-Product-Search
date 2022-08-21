import tensorflow as tf
import numpy as np
import math

class config:
    SEED=17
    N_CLASSES = 471
    ARC_FACE_M = 0.3
    EMB_DIM = 1024
    head = 'arcface'
    LR = 1e-4

# Modified version of ArcFace from original kernel to create this. Based on paper ElasticFace: Elastic Margin Loss for Deep Face Recognition (https://arxiv.org/pdf/2109.09416.pdf).

class ElasticArcFace(tf.keras.layers.Layer):
    def __init__(
        self,
        n_classes,
        s=30,
        mean=0.50,
        std=0.025,
        easy_margin=False,
        ls_eps=0.0,
        **kwargs
    ):

        super(ElasticArcFace, self).__init__(**kwargs)
        
        print(f'ElasticArcFace mean: {mean}, s: {std}')

        self.n_classes = n_classes
        self.s = s
        self.mean = mean
        self.std = std
        self.ls_eps = ls_eps

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'mean': self.mean,
            'std': self.std,
            'ls_eps': self.ls_eps
        })
        return config

    def build(self, input_shape):
        super(ElasticArcFace, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=None)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))

        m = tf.random.normal((tf.shape(y)[0], 1), mean=self.mean, stddev=self.std, seed=config.SEED)

        cos_m = tf.math.cos(m)
        sin_m = tf.math.sin(m)
        th = tf.math.cos(math.pi - m)
        mm = tf.math.sin(math.pi - m) * m
        
        phi = cosine * cos_m - sine * sin_m

        phi = tf.where(cosine > th, phi, cosine - mm)
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
            )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

def freeze_BN(model):
    # Unfreeze layers while leaving BatchNorm layers frozen
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

def get_model():    
    # with strategy.scope():
    margin = ElasticArcFace(
        n_classes = config.N_CLASSES, 
        s = 30, 
        mean = config.ARC_FACE_M,
        std=0.025,
        name=f'head/{config.head}', 
        dtype='float32'
    )
    inp = tf.keras.layers.Input(shape=[17, 768], name = 'inp1')
    label = tf.keras.layers.Input(shape=(), name = 'inp2')

    conv_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(1024, 3, activation='relu'),
        tf.keras.layers.Conv1D(2048, 3, activation='relu'),
        tf.keras.layers.Conv1D(1024, 3, activation='relu'),
    ])
    
    x = conv_model(inp)
    # Concat pooling
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    pretrained_out = tf.keras.layers.Concatenate()([avg_pool, max_pool])
    
    print(f'Size of embed {config.EMB_DIM}')
    pre_margin_dense_layer = tf.keras.layers.Dense(config.EMB_DIM)

    # Multiple-sample dropout https://arxiv.org/abs/1905.09788
    dropout_base = 0.17
    drop_ls = [tf.keras.layers.Dropout((dropout_base + 0.01), seed=420),
               tf.keras.layers.Dropout((dropout_base + 0.02), seed=4200),
               tf.keras.layers.Dropout((dropout_base + 0.03), seed=42000),
               tf.keras.layers.Dropout((dropout_base + 0.04), seed=420000),
               tf.keras.layers.Dropout((dropout_base + 0.05), seed=4200000)]
    for ii, drop in enumerate(drop_ls):
        if ii == 0:
            embed = (pre_margin_dense_layer(drop(pretrained_out)) / 5.0)
        else:
            embed += (pre_margin_dense_layer(drop(pretrained_out)) / 5.0)
            
    embed = tf.keras.layers.BatchNormalization()(embed)
    embed = tf.math.l2_normalize(embed, axis=1)
    
    x = margin([embed, label])
    output = tf.keras.layers.Softmax(dtype='float32', name='metric_out')(x)
    
    model = tf.keras.models.Model(inputs = [inp, label], outputs=output)
    embed_model = tf.keras.models.Model(inputs = inp, outputs = embed)  
    
    opt = tf.keras.optimizers.Adam(learning_rate = config.LR)
    model.compile(
        optimizer = opt,
        loss={'metric_out': tf.keras.losses.SparseCategoricalCrossentropy()},
        metrics={'metric_out': [tf.keras.metrics.SparseCategoricalAccuracy(), 
                                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=20)]}
    ) 
    
    return model, embed_model