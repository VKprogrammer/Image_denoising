import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose,\
                                    GlobalAveragePooling2D, AveragePooling2D, MaxPool2D, UpSampling2D,\
                                    BatchNormalization, Activation, Flatten, Dense, Input,\
                                    Add, Multiply, Concatenate, concatenate, Softmax
from tensorflow.keras import initializers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import softmax

tf.keras.backend.set_image_data_format('channels_last')



class Convolutional_block(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')
        self.conv_2 = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')
        self.conv_3 = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')
        self.conv_4 = Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same')

    def call(self, X):
        X_1 = self.conv_1(X)
        X_1 = Activation('relu')(X_1)

        X_2 = self.conv_2(X_1)
        X_2 = Activation('relu')(X_2)

        X_3 = self.conv_3(X_2)
        X_3 = Activation('relu')(X_3)

        # X_4 = self.conv_4(X_3)
        # X_4 = Activation('relu')(X_4)

        #print('---conv block=',X_4.shape)

        return X_3
    



class Channel_attention(tf.keras.layers.Layer):
    def __init__(self, C=64, **kwargs):
        super().__init__(**kwargs)
        self.C = C
        self.gap = GlobalAveragePooling2D()
        self.dense_middle = Dense(units=2, activation='relu')
        self.dense_sigmoid = Dense(units=self.C, activation='sigmoid')

    def get_config(self):
        config = super().get_config().copy()
        config.update({'C': self.C})
        return config

    def call(self, X):
        v = self.gap(X)
        fc1 = self.dense_middle(v)
        mu = self.dense_sigmoid(fc1)
        mu = tf.reshape(mu, [-1, 1, 1, self.C])  # Adjust shape for multiplication
        U_out = Multiply()([X, mu])
        return U_out



class Spatial_attention(tf.keras.layers.Layer):
    def _init_(self, **kwargs):
        super()._init_(**kwargs)
        self.conv1 = Conv2D(filters=1, kernel_size=(7, 7), padding='same', activation='sigmoid')

    def call(self, X):
        avg_pool = tf.reduce_mean(X, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(X, axis=-1, keepdims=True)
        concat = Concatenate()([avg_pool, max_pool])
        attention = self.conv1(concat)
        return Multiply()([X, attention])



class Avg_pool_Unet_Upsample_msfe(tf.keras.layers.Layer):
    def __init__(self, pool_size, upsample_rate, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.upsample_rate = upsample_rate
        self.avg_pool = AveragePooling2D(pool_size=pool_size, padding='same')
        self.upsample = UpSampling2D(upsample_rate, interpolation='bilinear')
        self.conv_3 = Conv2D(filters=3, kernel_size=[1, 1])

        # Initialize convolutional layers for U-Net
        self.conv_down_layers = []
        self.conv_up_layers = []

        filter_size = 64
        for i in range(4):
            self.conv_down_layers.append([
                Conv2D(filters=filter_size, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
                Conv2D(filters=filter_size, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))
            ])
            filter_size *= 2
            filter_size //= 2
        for i in range(4):
            self.conv_up_layers.append([
                Conv2DTranspose(filters=filter_size, kernel_size=(3, 3), strides=2, padding='same'),
                Conv2D(filters=filter_size, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
                Conv2D(filters=filter_size, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))
            ])
            filter_size //= 2

    def get_config(self):
        config = super().get_config().copy()
        config.update({'avg_pool_size': self.avg_pool_size, 'upsample_rate': self.upsample_rate})
        return config

    def call(self, X):
        # Apply average pooling
        X = self.avg_pool(X)

        # Downsampling path
        conv1 = X
        for conv in self.conv_down_layers:
            conv1 = conv[0](conv1)
            conv1 = conv[1](conv1)
            conv1 = MaxPool2D(pool_size=(2, 2), padding='same')(conv1)

        # Upsampling path
        for conv in self.conv_up_layers:
            conv1 = conv[0](conv1)
            conv1 = conv[1](conv1)
            conv1 = conv[2](conv1)

        # Final convolution and upsampling
        conv1 = self.conv_3(conv1)
        X = self.upsample(conv1)
        return X
    

class Multi_scale_feature_extraction(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.msfe_16 = Avg_pool_Unet_Upsample_msfe(pool_size=16, upsample_rate=16)
        self.msfe_8 = Avg_pool_Unet_Upsample_msfe(pool_size=8, upsample_rate=8)
        self.msfe_4 = Avg_pool_Unet_Upsample_msfe(pool_size=4, upsample_rate=4)
        self.msfe_2 = Avg_pool_Unet_Upsample_msfe(pool_size=2, upsample_rate=2)
        self.msfe_1 = Avg_pool_Unet_Upsample_msfe(pool_size=1, upsample_rate=1)

    def call(self, X):
        up_sample_16 = self.msfe_16(X)
        up_sample_8 = self.msfe_8(X)
        up_sample_4 = self.msfe_4(X)
        up_sample_2 = self.msfe_2(X)
        up_sample_1 = self.msfe_1(X)
        msfe_out = Concatenate()([X, up_sample_16, up_sample_8, up_sample_4,up_sample_2])

        #print('---Multi scale feature extraction block=',msfe_out.shape)
        return msfe_out
    

class Kernel_selecting_module(tf.keras.layers.Layer):
    def __init__(self, C=21, **kwargs):
        super().__init__(**kwargs)
        self.C = C
        self.c_3 = Conv2D(filters=self.C, kernel_size=(3,3), strides=1, padding='same', kernel_regularizer=regularizers.l2(l2=0.001))
        self.c_5 = Conv2D(filters=self.C, kernel_size=(5,5), strides=1, padding='same', kernel_regularizer=regularizers.l2(l2=0.001))
        self.c_7 = Conv2D(filters=self.C, kernel_size=(7,7), strides=1, padding='same', kernel_regularizer=regularizers.l2(l2=0.001))
        self.gap = GlobalAveragePooling2D()
        self.dense_two = Dense(units=2, activation='relu')
        self.dense_c1 = Dense(units=self.C)
        self.dense_c2 = Dense(units=self.C)
        self.dense_c3 = Dense(units=self.C)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'C': self.C
        })
        return config

    def call(self, X):
        X_1 = self.c_3(X)
        X_2 = self.c_5(X)
        X_3 = self.c_7(X)

        X_dash = Add()([X_1, X_2, X_3])

        v_gap = self.gap(X_dash)
        v_gap = tf.reshape(v_gap, [-1, 1, 1, self.C])
        fc1 = self.dense_two(v_gap)

        alpha = self.dense_c1(fc1)
        beta = self.dense_c2(fc1)
        gamma = self.dense_c3(fc1)

        before_softmax = concatenate([alpha, beta, gamma], 1)
        # print(before_softmax.shape)
        after_softmax = softmax(before_softmax, axis=1)
        a1 = after_softmax[:, 0, :, :]
        # print(a1)
        a1 = tf.reshape(a1, [-1, 1, 1, self.C])
        # print(a1)
        a2 = after_softmax[:, 1, :, :]
        a2 = tf.reshape(a2, [-1, 1, 1, self.C])
        a3 = after_softmax[:, 2, :, :]
        a3 = tf.reshape(a3, [-1, 1, 1, self.C])

        select_1 = Multiply()([X_1, a1])
        select_2 = Multiply()([X_2, a2])
        select_3 = Multiply()([X_3, a3])

        out = Add()([select_1, select_2, select_3])

        return out
    

def create_model():
    # ca_block = Channel Attention block
    # msfe_block = Multi scale feature extraction block
    # ksm = Kernel Selecting Module
    tf.keras.backend.clear_session()

    input = Input(shape=(256,256,3), name="input_layer")
    print("Input =",input.shape)

    conv_block = Convolutional_block()(input)
    print("Conv block =",conv_block.shape)
    ca_block = Channel_attention()(conv_block)
    print("Channel Attention =",ca_block.shape)
    ca_block = Conv2D(filters=3, kernel_size=(3,3), strides=1, padding='same')(ca_block)
    print("Channel Attention Last CNN =",ca_block.shape)
    ca_block = Concatenate()([input, ca_block])
    print("First phase =",ca_block.shape)
    print()

    msfe_block = Multi_scale_feature_extraction()(ca_block)

    print("Multi-scale feature extraction =",msfe_block.shape)

    ksm = Kernel_selecting_module()(msfe_block)
    ksm = Conv2D(filters=3, kernel_size=(3,3), strides=1, padding='same')(ksm)
    print("Kernel Selection Module =",ksm.shape)
    model = Model(inputs=[input], outputs=[ksm])
    return model

# model = create_model()
# model1=create_model()
# model1.summary()
# model.summary()