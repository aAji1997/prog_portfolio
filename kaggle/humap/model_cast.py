#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 13:10:40 2022

@author: hal
"""
import tensorflow as tf
import keras
import numpy as np

import tensorflow_lattice as tfl
import tensorflow_addons as tfa
import copy

'''
Helper Functions
'''
def get_norm_axes(inp):
    try:
        iterator = iter(inp)
        num_dims = len(inp)
        axes_to_get = [-i for i in range(1, num_dims + 1)]

        return axes_to_get

    except TypeError:
        axes_to_get = -1

        return axes_to_get

def recur_layer_init(layer):
    if isinstance(layer, tfl.layers.Linear):
        layer.kernel_initializer= keras.initializers.initializers_v2.GlorotUniform()
    if isinstance(layer, tfl.layers.Linear) and layer.use_bias is not False:
        layer.bias_initializer = keras.initializers.initializers_v2.RandomNormal(stddev=1e-6)
    if hasattr(layer, 'layers'):
        for l in layer.layers:
            recur_layer_init(l)
            
def res_init(layer):
    if isinstance(layer, keras.layers.Conv2D):
        layer.kernel_initializer= keras.initializers.initializers_v2.HeNormal()
        
    if hasattr(layer, 'layers'):
        for l in layer.layers:
            res_init(l)


'''
Preprocessing/Processing layers
'''
class Flatten(keras.layers.Layer):
    def __init__(self):
        super(Flatten, self).__init__()
        self.trainable = False
    
    def call(self, x):
        return tf.reshape(tensor=x, shape=(x.shape[0], -1))
    
    
class Augment(keras.layers.Layer):

    def __init__(self, augs, seed):
        super().__init__()
        self.trainable= False
        
        self.augs = augs
        self.seed = seed
        self.aug_dict = {
            'flip': keras.layers.RandomFlip,
            'rotate': keras.layers.RandomRotation,
            'contrast': keras.layers.RandomContrast,
            'zoom': keras.layers.RandomZoom,
            'translate': keras.layers.RandomTranslation
        }
        self.aug_params = {
            "flip": {"mode": "horizontal_and_vertial", "seed": self.seed},
            "rotate": {"factor": 0.3, "fill_mode": "nearest", "interpolation": "nearest", "seed": self.seed},
            "contrast": {"factor": 0.2, "seed": self.seed},
            "zoom": {"height_factor": 0.3, "width_factor": 0.3, "fill_mode": "nearest", "seed": self.seed},
            "translate": {"height_factor": 0.3, "width_factor": 0.3, "fill_mode": "nearest", "seed": self.seed}

        }
        self.available_augs = set(self.aug_dict.keys())
        self.given_augs = set(self.augs)
        self.applied_augs = self.available_augs.intersection(self.given_augs)
        assert len(
            self.applied_augs) == 0, f"No augmentation matching: {self.given_augs}, choose one of: {self.available_augs}"

        self.applied_aug_dict = {k: v for k, v in self.aug_dict.items() if k in self.applied_augs}
        self.applied_param_dict = {k: v for k, v in self.aug_params.items() if k in self.applied_augs}
        self.aug_network = keras.Sequential(
            [self.applied_aug_dict[key](**keypars) for key, keypars in self.applied_param_dict.items()])

    def call(self, images, masks):
        images = tf.transpose(images, perm=(0, 3, 2, 1))
        masks = tf.transpose(masks, perm=(0, 3, 2, 1))
        
        images = self.aug_network(images)
        masks = self.aug_network(masks)
        
        images = tf.transpose(images, perm=(0, 3, 2, 1))
        masks = tf.transpose(masks, perm=(0, 3, 2, 1))
        
        return images, masks
    
'''
Basic Transformer scaffolding section
'''
        
class SpatialEmbeddings(keras.Model):
    """
    Construct the embeddings from patch and position embeddings
    """
    def __init__(self, config, patchsize, img_size, in_channels, **kwargs):
        super().__init__(**kwargs)
        img_size = (img_size, img_size)
        patch_size = (patchsize, patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = keras.layers.Conv2D(filters=config.transformer["embedding_channels"],
                                                    kernel_size=patch_size, strides=patch_size, data_format="channels_first")
        self.position_embeddings = tf.Variable(tf.zeros((1, n_patches, config.transformer["embedding_channels"])))
        self.flatten = keras.layers.Flatten(data_format="channels_first")

    def call(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)
        x = self.flatten(x)
        x = tf.experimental.numpy.swapaxes(x, -1, -2)
        embeddings = x + self.position_embeddings
        return embeddings


class Attention(keras.Model):
    """
    Dispersed attention module, distribute transformer attention accross global range for a sample, attending to local features whilst linking local features
    and maintaining a progressively improving global featurespace representation
    """

    def __init__(self, config, channel_num, **kwargs):
        super().__init__(**kwargs)
        self.KV_size = config.KV_size_S
        self.KV_size_C = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = self.KV_size // self.num_attention_heads

        self.query1 = tfl.layers.Linear(num_input_dims=config.transformer["embedding_channels"],
                                        units=config.transformer["embedding_channels"], use_bias=False)
        self.query2 = tfl.layers.Linear(num_input_dims=config.transformer["embedding_channels"],
                                        units=config.transformer["embedding_channels"], use_bias=False)
        self.query3 = tfl.layers.Linear(num_input_dims=config.transformer["embedding_channels"],
                                        units=config.transformer["embedding_channels"], use_bias=False)
        self.query4 = tfl.layers.Linear(num_input_dims=config.transformer["embedding_channels"],
                                        units=config.transformer["embedding_channels"], use_bias=False)
        self.key = tfl.layers.Linear(num_input_dims=config.transformer["embedding_channels"],
                                     units=config.transformer["embedding_channels"], use_bias=False)
        self.value = tfl.layers.Linear(num_input_dims=config.transformer["embedding_channels"],
                                       units=config.transformer["embedding_channels"], use_bias=False)
        self.query_C = tfl.layers.Linear(num_input_dims=self.KV_size_C, units=self.KV_size_C, use_bias=False)
        self.key_C = tfl.layers.Linear(num_input_dims=self.KV_size_C, units=self.KV_size_C, use_bias=False)
        self.value_C = tfl.layers.Linear(num_input_dims=self.KV_size_C, units=self.KV_size_C, use_bias=False)
        self.psi1 = tfa.layers.InstanceNormalization(axis=1)
        self.psi2 = tfa.layers.InstanceNormalization(axis=1)
        self.softmax = keras.layers.Softmax()
        self.attn_norm = keras.layers.LayerNormalization(axis=get_norm_axes(config.KV_size_S), epsilon=1e-6)

        self.out1 = tfl.layers.Linear(num_input_dims=config.transformer["embedding_channels"],
                                      units=config.transformer["embedding_channels"], use_bias=False)
        self.out2 = tfl.layers.Linear(num_input_dims=config.transformer["embedding_channels"],
                                      units=config.transformer["embedding_channels"], use_bias=False)
        self.out3 = tfl.layers.Linear(num_input_dims=config.transformer["embedding_channels"],
                                      units=config.transformer["embedding_channels"], use_bias=False)
        self.out4 = tfl.layers.Linear(num_input_dims=config.transformer["embedding_channels"],
                                      units=config.transformer["embedding_channels"], use_bias=False)
        self.attn_dropout = keras.layers.Dropout(rate=config.transformer["attention_dropout_rate"])
        self.proj_dropout = keras.layers.Dropout(rate=config.transformer["attention_dropout_rate"])
        self.permute = keras.layers.Permute(dims=(2, 1, 3))

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = tf.reshape(tensor=x, shape=new_x_shape)
        return self.permute(x)

    def call(self, emb1, emb2, emb3, emb4, emb_C):
        Q_C = self.query_C(emb_C)
        K_C = self.key_C(emb_C)
        V_C = self.value_C(emb_C)

        attn = tf.matmul(tf.experimental.numpy.swapaxes(Q_C, -1, -2), K_C)
        attn = tf.expand_dims(attn, axis=1)
        ch_similarity_matrix = tf.squeeze(self.softmax(self.psi1(attn)), axis=1)
        ch_similarity_matrix = self.attn_dropout(ch_similarity_matrix)
        context_layer = tf.matmul(ch_similarity_matrix, tf.experimental.numpy.swapaxes(V_C, -1, -2))
        T_hat = (tf.experimental.numpy.swapaxes(context_layer, -1, -2))
        KV_S = tf.split(value=T_hat, num_or_size_splits=self.KV_size_C // 4, axis=2)
        KV_S = tf.concat(KV_S, axis=1)

        Q1 = self.query1(emb1)
        Q2 = self.query2(emb2)
        Q3 = self.query3(emb3)
        Q4 = self.query4(emb4)
        K = self.key(KV_S)
        V = self.value(KV_S)

        multi_head_Q1 = self.transpose_for_scores(Q1)
        multi_head_Q2 = self.transpose_for_scores(Q2)
        multi_head_Q3 = self.transpose_for_scores(Q3)
        multi_head_Q4 = self.transpose_for_scores(Q4)
        multi_head_K = tf.experimental.numpy.swapaxes(a=self.transpose_for_scores(K), axis1=-1, axis2=-2)
        multi_head_V = self.transpose_for_scores(V)

        attn1 = tf.matmul(multi_head_Q1, multi_head_K)
        attn2 = tf.matmul(multi_head_Q2, multi_head_K)
        attn3 = tf.matmul(multi_head_Q3, multi_head_K)
        attn4 = tf.matmul(multi_head_Q4, multi_head_K)

        sp_similarity_matrix1 = self.softmax(self.psi2(attn1))
        sp_similarity_matrix2 = self.softmax(self.psi2(attn2))
        sp_similarity_matrix3 = self.softmax(self.psi2(attn3))
        sp_similarity_matrix4 = self.softmax(self.psi2(attn4))

        sp_similarity_matrix1 = self.attn_dropout(sp_similarity_matrix1)
        sp_similarity_matrix2 = self.attn_dropout(sp_similarity_matrix2)
        sp_similarity_matrix3 = self.attn_dropout(sp_similarity_matrix3)
        sp_similarity_matrix4 = self.attn_dropout(sp_similarity_matrix4)

        context_layer1 = tf.matmul(sp_similarity_matrix1, multi_head_V)
        context_layer2 = tf.matmul(sp_similarity_matrix2, multi_head_V)
        context_layer3 = tf.matmul(sp_similarity_matrix3, multi_head_V)
        context_layer4 = tf.matmul(sp_similarity_matrix4, multi_head_V)

        context_layer1 = self.permute(context_layer1)
        new_context_layer_shape = context_layer1.shape[:-2] + tuple(self.KV_size)
        context_layer1 = tf.reshape(context_layer1, new_context_layer_shape)
        context_layer2 = tf.reshape(context_layer2, new_context_layer_shape)
        context_layer3 = tf.reshape(context_layer3, new_context_layer_shape)
        context_layer4 = tf.reshape(context_layer4, new_context_layer_shape)
        
        Q1 = self.out1(context_layer1)
        Q2 = self.out2(context_layer2)
        Q3 = self.out3(context_layer3)
        Q4 = self.out4(context_layer4)

        Q1 = self.proj_dropout(Q1)
        Q2 = self.proj_dropout(Q2)
        Q3 = self.proj_dropout(Q3)
        Q4 = self.proj_dropout(Q4)

        return Q1, Q2, Q3, Q4


class Mlp(keras.Model):
    def __init__(self, config, in_channel, mlp_channel):
        super(Mlp, self).__init__()
        self.act_fn = keras.activations.gelu
        self.dropout = keras.layers.Dropout(rate=config.transformer["dropout_rate"])
        self.glorot = keras.initializers.initializers_v2.GlorotUniform()
        self.normal = keras.initializers.initializers_v2.RandomNormal(stddev=1e-6)

        self.fc1 = tfl.layers.Linear(num_input_dims=in_channel, units=mlp_channel, kernel_initializer=self.glorot,
                                     bias_initializer=self.normal)
        self.fc2 = tfl.layers.Linear(num_input_dims=mlp_channel, units=in_channel, kernel_initializer=self.glorot,
                                     bias_initializer=self.normal)

    def call(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block_ViT(keras.Model):
    """
    Vision transformer block
    """

    def __init__(self, config, channel_num, **kwargs):
        super(Block_ViT, self).__init__()
        expand_ratio = config.expand_ratio
        self.attn_norm1 = keras.layers.LayerNormalization(axis=get_norm_axes(config.transformer["embedding_channels"]),
                                                          epsilon=1e-6)
        self.attn_norm2 = keras.layers.LayerNormalization(axis=get_norm_axes(config.transformer["embedding_channels"]),
                                                          epsilon=1e-6)
        self.attn_norm3 = keras.layers.LayerNormalization(axis=get_norm_axes(config.transformer["embedding_channels"]),
                                                          epsilon=1e-6)
        self.attn_norm4 = keras.layers.LayerNormalization(axis=get_norm_axes(config.transformer["embedding_channels"]),
                                                          epsilon=1e-6)

        self.attn_norm = keras.layers.LayerNormalization(axis=get_norm_axes(config.KV_size_S), epsilon=1e-6)
        self.attn_norm_C = keras.layers.LayerNormalization(axis=get_norm_axes(config.KV_size), epsilon=1e-6)
        self.channel_attn = Attention(config=config, channel_num=channel_num)

        self.ffn_norm1 = keras.layers.LayerNormalization(axis=get_norm_axes(config.transformer["embedding_channels"]),
                                                         epsilon=1e-6)
        self.ffn_norm2 = keras.layers.LayerNormalization(axis=get_norm_axes(config.transformer["embedding_channels"]),
                                                         epsilon=1e-6)
        self.ffn_norm3 = keras.layers.LayerNormalization(axis=get_norm_axes(config.transformer["embedding_channels"]),
                                                         epsilon=1e-6)
        self.ffn_norm4 = keras.layers.LayerNormalization(axis=get_norm_axes(config.transformer["embedding_channels"]),
                                                         epsilon=1e-6)

        self.ffn1 = Mlp(config, config.transformer["embedding_channels"],
                        config.transformer["embedding_channels"] * expand_ratio)
        self.ffn2 = Mlp(config, config.transformer["embedding_channels"],
                        config.transformer["embedding_channels"] * expand_ratio)
        self.ffn3 = Mlp(config, config.transformer["embedding_channels"],
                        config.transformer["embedding_channels"] * expand_ratio)
        self.ffn4 = Mlp(config, config.transformer["embedding_channels"],
                        config.transformer["embedding_channels"] * expand_ratio)

    def call(self, emb1, emb2, emb3, emb4):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3
        org4 = emb4

        for i in range(4):
            var_name = "emb" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        emb_C = tf.concat(embcat, axis=2)
        cx1 = self.attn_norm1(emb1)
        cx2 = self.attn_norm2(emb2)
        cx3 = self.attn_norm3(emb3)
        cx4 = self.attn_norm4(emb4)
        emb_C = self.attn_norm_C(emb_C)
        cx1, cx2, cx3, cx4 = self.channel_attn(cx1, cx2, cx3, cx4, emb_C)
        cx1 = org1 + cx1
        cx2 = org2 + cx2
        cx3 = org3 + cx3
        cx4 = org4 + cx4

        org1 = cx1
        org2 = cx2
        org3 = cx3
        org4 = cx4

        x1 = self.ffn_norm1(cx1)
        x2 = self.ffn_norm2(cx2)
        x3 = self.ffn_norm3(cx3)
        x4 = self.ffn_norm4(cx4)

        x1 = self.ffn1(x1)
        x2 = self.ffn2(x2)
        x3 = self.ffn3(x3)
        x4 = self.ffn4(x4)

        x1 = x1 + org1
        x2 = x2 + org2
        x3 = x3 + org3
        x4 = x4 + org4

        return x1, x2, x3, x4


class Encoder(keras.Model):
    def __init__(self, config, channel_num, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.layer = []
        self.encoder_norm1 = keras.layers.LayerNormalization(axis=get_norm_axes(config.transformer["embedding_channels"]), epsilon=1e-6)
        self.encoder_norm2 = keras.layers.LayerNormalization(axis=get_norm_axes(config.transformer["embedding_channels"]), epsilon=1e-6)
        self.encoder_norm3 = keras.layers.LayerNormalization(axis=get_norm_axes(config.transformer["embedding_channels"]), epsilon=1e-6)
        self.encoder_norm4 = keras.layers.LayerNormalization(axis=get_norm_axes(config.transformer["embedding_channels"]), epsilon=1e-6)
        
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT(config=config, channel_num=channel_num)
            self.layer.append(copy.deepcopy(layer))
            
    def call(self, emb1, emb2, emb3, emb4):
        for layer_block in self.layer:
            emb1, emb2, emb3, emb4 = layer_block(emb1, emb2, emb3, emb4)
        emb1 = self.encoder_norm1
        emb2 = self.encoder_norm2(emb2)
        emb3 = self.encoder_norm3(emb3)
        emb4 = self.encoder_norm4(emb4)
        
        return emb1, emb2, emb3, emb4
    
'''
Distributed Attention Transformer section
'''
    
class DAT(keras.Model):
    def __init__(self, config, img_size, channel_num, patchSize, **kwargs):
        super(DAT, self).__init__(**kwargs)
        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]
        
        self.embeddings_1 = SpatialEmbeddings(config=config, patchsize=self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddings_2 = SpatialEmbeddings(config=config, patchsize=self.patchSize_2, img_size=img_size //2, in_channels=channel_num[1])
        self.embeddings_3 = SpatialEmbeddings(config=config, patchsize=self.patchSize_3, img_size=img_size // 4, in_channels=channel_num[2])
        self.embeddings_4 = SpatialEmbeddings(config=config, patchsize=self.patchSize_4, img_size=img_size // 8, in_channels=channel_num[3])
        self.encoder = Encoder(config=config, channel_num=channel_num)
        self._init_weights()
        
    def _init_weights(self):
        for layer in self.layers:
            recur_layer_init(layer)
            
    def call(self, en1, en2, en3, en4):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)
        enc1, enc2, enc3, enc4 = self.encoder(emb1, emb2, emb3, emb4)
        
        return enc1, enc2, enc3, enc4
    
class Reconstruct(keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, **kwargs):
        super(Reconstruct, self).__init__(**kwargs)
        
        if kernel_size == 3:
            padding = "same"
        else:
            padding = "valid"
        
        self.conv = keras.layers.Conv2D(filters=out_channels, kernel_size=kernel_size, padding=padding, data_format="channels_first")
        self.norm = keras.layers.BatchNormalization(axis=1)
        self.activation = keras.layers.ReLU()
        self.scale_factor = scale_factor
        self.upsample = keras.layers.UpSampling2D(size=self.scale_factor, data_format="channels_first")
        self.permute = keras.layers.Permute(dims=(2, 1))
        
    def call(self, x):
        B, n_patch, hidden = x.shape
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = self.permute(x)
        x = tf.reshape(tensor=x, shape=(B, hidden, h, w))
        
        if self.scale_factor[0] > 1:
            x = self.upsample(x)
        
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        
        return out

class DownBlock(keras.Model):
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.Maxpool = keras.layers.MaxPool2D(pool_size=2, strides=2, data_format="channels_first")
        self.conv = keras.Sequential(
            [keras.layers.Conv2D(filters=out_ch, kernel_size=3, strides=1, padding="same", data_format="channels_first", use_bias=True), 
             keras.layers.BatchNormalization(axis=1), keras.layers.ReLU(), 
             keras.layers.Conv2D(filters=out_ch, kernel_size=3, strides=1, padding="same", data_format="channels_first", use_bias=True),
             keras.layers.BatchNormalization(axis=1), keras.layers.ReLU() 
             ]
            )
        
    def call(self, x):
        x = self.Maxpool(x)
        x = self.conv(x)
        return x
    
class DRA_C(keras.Model):
    '''
    Channel-wise DRA Module
    '''
    def __init__(self, skip_dim, decoder_dim, img_size, config, **kwargs):
        super(DRA_C, self).__init__(**kwargs)
        self.patch_size = img_size // 14
        self.ft_size = img_size
        self.patch_embeddings = keras.layers.Conv2D(filters=decoder_dim, kernel_size=self.patch_size, strides=self.patch_size, data_format="channels_first", use_bias=True)
        self.conv = keras.Sequential([keras.layers.Conv2D(filters=skip_dim, kernel_size=(1, 1), strides=1,  data_format="channels_first", use_bias=True),
                                      keras.layers.BatchNormalization(axis=1), keras.layers.ReLU()
                                      ])
        self.query = tfl.layers.Linear(num_input_dims=decoder_dim, units=skip_dim, use_bias=False)
        self.key = tfl.layers.Linear(num_input_dims=config.transformer.embedding_channels, units=skip_dim, use_bias=False)
        self.value = tfl.layers.Linear(num_input_dims=config.transformer.embedding_channels, units=skip_dim, use_bias=False)
        self.out = tfl.layers.Linear(num_input_dims=skip_dim, units=skip_dim, use_bias=False)
        self.softmax = keras.layers.Softmax()
        self.psi = tfa.layers.InstanceNormalization(axis=1)
        self.relu = keras.layers.ReLU()
        self.reconstruct = Reconstruct(in_channels=skip_dim, out_channels=skip_dim, kernel_size=1, scale_factor=(self.patch_size, self.patch_size))
        
    def call(self, decoder, trans):
        decoder_mask = self.conv(decoder)
        decoder_L = self.patch_embeddings(decoder)
        decoder_L = tf.reshape(tensor=decoder_L, shape=(*decoder_L.shape[:2], -1))
        decoder_L = tf.transpose(decoder_L, perm=(-1, -2))
        query = tf.transpose(self.query(decoder_L), perm=(-1, -2))
        key = self.key(trans)
        value = tf.transpose(self.value(trans), perm=(-1, -2))
        ch_similarity_matrix = tf.matmul(query, key)
        ch_similarity_matrix = self.softmax(tf.squeeze(self.psi(tf.expand_dims(ch_similarity_matrix, axis=1)), axis=1))
        out = tf.transpose(tf.matmul(ch_similarity_matrix, value), perm=(-1, -2))
        out = self.out(out)
        out = self.reconstruct(out)
        out = out * decoder_mask
        return out
    
class DRA_S(keras.Model):
    '''
    Spatial-wise DRA module
    '''
    def __init__(self, skip_dim, decoder_dim, img_size, config, **kwargs):
        super(DRA_S, self).__init__(**kwargs)
        self.patch_size = img_size // 14
        self.ft_size = img_size
        self.patch_embeddings = keras.layers.Conv2D(filters=decoder_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.conv = keras.Sequential([keras.layers.Conv2D(filters=skip_dim, kernel_size=(1, 1), use_bias=True), 
                                      keras.layers.BatchNormalization(axis=1), keras.layers.ReLU()
                                      ])
        self.query = tfl.layers.Linear(num_input_dims=decoder_dim, units=skip_dim, use_bias=False)
        self.key = tfl.layers.Linear(num_input_dims=config.transformer.embedding_channels, units=skip_dim, use_bias=False)
        self.value = tfl.layers.Linear(num_input_dims=config.transformer.embedding_channels, units=skip_dim, use_bias=False)
        self.out = tfl.layers.Linear(num_input_dims=skip_dim, units=skip_dim, use_bias=False)
        self.softmax = keras.layers.Softmax()
        self.psi = tfa.layers.InstanceNormalization(axis=1)
        self.reconstruct = Reconstruct(skip_dim, skip_dim, kernel_size=1, scale_factor=(self.patch_size, self.patch_size))
        
    def call(self, decoder, trans):
        decoder_mask = self.conv(decoder)
        decoder_L = self.patch_embeddings(decoder)
        decoder_L = tf.reshape(tensor=decoder, shape=(*decoder_L.shape[:2], -1))
        decoder_L = tf.transpose(decoder_L, perm=(-1, -2))
        query = self.query(decoder_L)
        key = tf.transpose(self.key(trans), perm=(-1, -2))
        value = self.value(trans)
        sp_similarity_matrix = tf.matmul(query, key)
        sp_similarity_matrix = self.softmax(tf.squeeze(self.psi(tf.expand_dims(sp_similarity_matrix, axis=0)), axis=0))
        out = tf.matmul(sp_similarity_matrix, value)
        out = self.out(out)
        out = self.reconstruct(out)
        out = out * decoder_mask
        return out
    
class UpBlock(keras.Model):
    def __init__(self, in_ch, skip_ch, out_ch, img_size, config):
        super(UpBlock, self).__init__()
        self.scale_factor = (img_size // 14, img_size // 14)
        self.up = keras.Sequential(
            [keras.layers.Conv2DTranspose(filters=in_ch // 2, kernel_size=2, strides=2, data_format="channels_first"),
             keras.layers.BatchNormalization(axis=1), keras.layers.ReLU()
             ]
            )
        self.pam = DRA_C(skip_dim=skip_ch, decoder_dim=in_ch // 2, img_size=img_size, config=config) #channel_wise_DRA
        #self.pam = DRA_S(skip_ch, in_ch//2, img_size, config) #spatial_wise_DRA
        self.conv = keras.Sequential([
            keras.layers.Conv2D(filters=out_ch, kernel_size=3, strides=1, padding="same", data_format="channels_first", use_bias=True),
            keras.layers.BatchNormalization(axis=1), keras.layers.ReLU(), 
            keras.layers.Conv2D(filters=out_ch, kernel_size=3, strides=1, padding="same", data_format="channels_first", use_bias=True),
            keras.layers.BatchNormalization(axis=1), keras.layers.ReLU()
            ])
        
    def call(self, decoder, o_i):
        d_i = self.up(decoder)
        o_hat_i = self.pam(d_i, o_i)
        x = tf.concat(values=(o_hat_i, d_i), axis=1)
        x = self.conv(x)
        return x
    
'''
Residual Network Section
'''
class ResBlock(keras.Model):
    expansion = 1
    
    def conv3x3(self):
        return keras.layers.Conv2D(filters=self.planes, kernel_size=self.kernel_size, strides=self.stride, groups=self.groups, 
                                   padding="same", dilation_rate=self.dilation, use_bias=False, data_format="channels_first", )
    
    def __init__(self, planes, stride=1, kernel_size=3, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        
        self.planes = planes
        self.stride = stride
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.groups = groups
        self.base_width = base_width
        self.dilation = dilation
        assert self.groups==1 or self.base_width ==64, f"Only groups = 1 and base_width = 64 allowed in ResBlock, given {self.groups} groups and {self.base_width} base_width" 
        assert self.dilation==1, "Dilation > 1 Not allowed in ResBlock"
        
        self.norm_layer = keras.layers.BatchNormalization(axis=1) if norm_layer is None else norm_layer
        self.conv1 = self.conv3x3()
        self.bn1 = self.norm_layer
        self.relu = keras.layers.ReLU()
        self.conv2 = self.conv3x3()
        self.bn2 = self.norm_layer


    
    def call(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out
    
class Bottleneck(keras.Model):
    def __init__(self, planes, stride=1, kernel_size=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        
        self.planes = planes
        self.stride = stride
        self.kernel_size = kernel_size
        self.downsample = downsample
        self.groups = groups
        self.base_width = base_width
        self.dilation = dilation
        
        self.norm_layer = keras.layers.BatchNormalization(axis=1) if norm_layer is None else norm_layer
        self.width = int(self.planes * (self.base_width / 64.0)) * groups
        self.conv1 = self.conv1x1()
        self.bn1 = self.norm_layer
        self.conv2 = self.conv3x3()
        self.bn2 = self.norm_layer
        self.conv3 = self.conv1x1()
        self.bn3 = norm_layer
        self.relu = keras.layers.ReLU()
        
    def conv1x1(self):
        return keras.layers.Conv2D(filters=self.width, kernel_size=self.kernel_size, strides=self.stride, use_bias=False,
                                   data_format="channel_first")
    
    def conv3x3(self):
        return keras.layers.Conv2D(filters=self.planes, kernel_size=self.kernel_size, strides=self.stride, groups=self.groups, 
                                   padding="same", use_bias=False, data_format="channels_first")
    
    def call(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(keras.Model):
    def __init__(self, block, layers, num_classes=100, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, **kwargs):
        super().__init__(**kwargs)
        if norm_layer is None:
            norm_layer = keras.layers.BatchNormalization(axis=1)
        self.norm_layer = norm_layer
        
        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = keras.layers.Conv2D(filters=self.inplanes, kernel_size=7, strides=2, padding="same", data_format="channels_first", use_bias=False)
        self.bn1 = self.norm_layer
        self.relu = keras.layers.ReLU()
        self.maxpool = keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same", data_format="channels_first")
        
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        self.avgpool = tfa.layers.AdaptiveAveragePooling2D(output_size=(1, 1), data_format="channels_first")
        self.fc = tfl.layers.Linear(num_input_dims=512 * block.expansion, units=num_classes)
        
        for layer in self.layers:
            res_init(layer)
        
    def conv1x1(self, filters, kernel_size, strides, padding="valid", data_format="channels_first", use_bias=False):
         return keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
                                          use_bias=use_bias)
     
    def make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = keras.Sequential([
                self.conv1x1(self.inplanes, planes * block.expansion, strides=stride),
                norm_layer
                ])
        
        layers = []
        layers.append(block(planes=planes, stride=stride, downsample=downsample, groups=self.groups, base_width=self.base_width, 
                            dilation=previous_dilation, norm_layer=norm_layer))
        
        for _ in range(1, blocks):
            layers.append(block(planes=planes,  groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        
        return keras.Sequential(layers=layers)
    
    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = tf.reshape(tensor=x, shape=(*x.shape[:1], -1))
        x = self.fc(x)
        
        return x
    
def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    
    return model

def resnet34():
    return _resnet(block=ResBlock, layers=[3, 4, 6, 3])

'''
Full UD Transformer implementation
'''    
class UDTransNet(keras.Model):
    def __init__(self, config, n_channels, n_classes, img_size, **kwargs):
        super(UDTransNet, self).__init__(**kwargs)
        self.n_classes = n_classes
        resnet = resnet34()
        filters_resnet = [64, 64, 128, 256, 512]
        filters_decoder = config.decoder_channels
        
        #Encoder
        self.Conv1 = keras.Sequential([
            keras.layers.Conv2D(filters=filters_resnet[0], kernel_size=3, strides=1, use_bias=True, data_format="channels_first"),
            keras.layers.BatchNormalization(axis=1),
            keras.layers.ReLU()
            ])
        self.Maxpool = keras.layers.MaxPool2D(pool_size=2, strides=2, data_format="channels_first")
        self.Conv2 = resnet.layer1
        self.Conv3 = resnet.layer2
        self.Conv4 = resnet.layer3
        self.Conv5 = resnet.layer4
        
        #DAT module
        self.mtc = DAT(config=config, img_size=img_size, channel_num=filters_resnet[0:4], patchSize=config.patch_sizes)
        
        #DRA and Decoder
        self.Up5 = UpBlock(in_ch=filters_resnet[4], skip_ch=filters_resnet[3], out_ch=filters_decoder[3], img_size=28, config=config)
        self.Up4 = UpBlock(in_ch=filters_decoder[3], skip_ch=filters_resnet[2], out_ch=filters_decoder[2], img_size=56, config=config)
        self.Up3 = UpBlock(in_ch=filters_decoder[2], skip_ch=filters_resnet[1], out_ch=filters_decoder[1], img_size=112, config=config)
        self.Up2 = UpBlock(in_ch=filters_decoder[1], skip_ch=filters_resnet[0], out_ch=filters_decoder[0], img_size=224, config=config)
        
        self.pred = keras.Sequential([
            keras.layers.Conv2D(filters=filters_decoder[0] // 2, kernel_size=1, data_format="channels_first"),
            keras.layers.BatchNormalization(axis=1),
            keras.layers.ReLU(),
            keras.layers.Conv2D(filters=n_classes, kernel_size=1, data_format="channels_first")
            ])
        self.last_activation = keras.activations.sigmoid
        
    def call(self, x):
        e1 = self.Conv1(x)
        e1_maxp = self.Maxpool(e1)
        e2 = self.Conv2(e1_maxp)
        e3 = self.Conv3(e2)
        e4 = self.Conv4(e3)
        e5 = self.Conv5(e4)
        
        o1, o2, o3, o4 = self.mtc(e1, e2, e3, e4)
        
        d4 = self.Up5(e5, o4)
        d3 = self.Up4(d4, o3)
        d2 = self.Up3(d3, o2)
        d1 = self.up2(d2, o1)
        
        if self.n_classes == 1:
            out = self.last_activation(self.pred(d1))
            
        else:
            out = self.pred(d1)
            
        return out
    

        
    
    

        
                                    
                                       
        
            
            
        
        