#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 00:08:26 2022

@author: hal
"""
import numpy
import tensorflow as tf
import tensorflow_io as tfio

import os
from keras.utils import save_img

import glob
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import tifffile as tiff

from tqdm import tqdm
from sklearn.model_selection import KFold
decode_tiff = tfio.experimental.image.decode_tiff
AUTO = tf.data.AUTOTUNE

def rechannel(img):
    img = img.transpose(2, 1, 0)
    return img

def rechannel_tf(img):
    img = tf.transpose(img, perm=(2, 1, 0))
    return img

def get_orig_mask(mask):
    mask = rechannel_tf(mask)
    new_mask = tf.image.resize(mask, (3000, 3000),
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    new_mask = tf.cast(new_mask, dtype=tf.float32)
    #new_mask = tf.transpose(new_mask, perm=(2, 1, 0))
    return new_mask

class tissue_dataset:
    def __init__(self, train_csv, images_per_shard, target_width=224, k_folds=7):
        self.OUTPUT_DIR = "./tissue-records"
        self.OUTPUT_NAME_PREFIX = "tissues_"
        self.TARGET_WIDTH = target_width
        self.train_csv = train_csv
        self.IMAGES_PER_SHARD = images_per_shard
        self.df = pd.read_csv(train_csv)
        self.image_ids = self.df['id']
        self.image_files = glob.glob("./train_images/*")
        
        self.num_folds = k_folds
        self.kfold = KFold(n_splits=self.num_folds, shuffle=True)

    def encode_rle(self, row):
        rle_enc = str(row['rle']).encode('utf-8')
        return rle_enc

    def mask2rle(self, img):
        pixels = img.T.flatten()
        pixels = np.pad(pixels, ((1, 1), ))
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)
 

    def rle2mask(self, mask_rle, shape=(3000, 3000)):
        
        shape = tf.convert_to_tensor(shape, tf.int64)
        size = tf.math.reduce_prod(shape)
        # Split string
        s = tf.strings.split(mask_rle)
        s = tf.strings.to_number(s, tf.int64)
        # Get starts and lengths
        starts = s[::2] - 1
        lens = s[1::2]
        # Make ones to be scattered
        total_ones = tf.reduce_sum(lens)
        ones = tf.ones([total_ones], tf.uint8)
        # Make scattering indices
        r = tf.range(total_ones)
        lens_cum = tf.math.cumsum(lens)
        s = tf.searchsorted(lens_cum, r, 'right')
        idx = r + tf.gather(starts - tf.pad(lens_cum[:-1], [(1, 0)]), s)
        # Scatter ones into flattened mask
        mask_flat = tf.scatter_nd(tf.expand_dims(idx, 1), ones, [size])
        # Reshape into mask
        return tf.transpose(tf.reshape(mask_flat, shape))
    
    def mask2rle_test(self, img):
        '''
        img: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        pixels= img.T.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)
     
    def rle2mask_test(self, mask_rle, shape=(3000, 3000)):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (width,height) of array to return 
        Returns numpy array, 1 - mask, 0 - background
    
        '''
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape).T

    def save_masks(self):
        for id, img in tqdm(zip(self.image_ids, self.image_files), total=len(self.image_ids)):
            img = tiff.imread(img)
            mask = self.rle2mask(self.df[self.df["id"] == id]["rle"].iloc[-1], (img.shape[1], img.shape[0]))
            mask = np.expand_dims(mask, axis=2)

            save_img(path=f"./train_masks/{id}.jpg", x=mask, data_format="channels_last")

        print("Finished Saving Masks")

    def show_example(self):
        print(self.df.dtypes)
        # print(len(self.df['rle'].iloc[0]))
        img_id_1 = 29223 #np.random.choice(self.df['id'].values)
        img_1 = tiff.imread("./train_images/" + str(img_id_1) + ".tiff")
        
        check_rle = self.df[self.df["id"]==img_id_1]["rle"].iloc[-1]
        
        mask_1 = self.rle2mask(self.df[self.df["id"]==img_id_1]["rle"].iloc[-1], (img_1.shape[1], img_1.shape[0]))
        organ_1 = self.df[self.df["id"] == img_id_1]["organ"].values
        print(img_1.shape)
        print(mask_1.shape)
        print(mask_1.dtype)
        print(organ_1)
        rle_1 = self.mask2rle(mask_1.numpy())
        #print(rle_1)
        
        assert check_rle == rle_1, "Mask conversion error"

        plt.figure()
        plt.imshow(img_1)
        plt.axis("off")

        plt.imshow(mask_1, cmap='coolwarm', alpha=0.5)
        plt.axis("off")

        plt.figure()
        plt.imshow(img_1)
        plt.imshow(mask_1, cmap='coolwarm', alpha=0.5)
        plt.axis("off")

    def read_and_decode(self, metadata):
        # print(tf.make_ndarray(metadata['fname']))
        img = tf.io.read_file(metadata['fname'])
        img = decode_tiff(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        image_size = tf.stack([metadata['width'], metadata['height']])
        image_size = tf.cast(image_size, dtype=tf.int32)

        mask = metadata['mname']
        ID = metadata['id']
        organ = metadata['organ']
        org_id = metadata['org_id']
        sex_id = metadata['sex_id']
        age = metadata['age']

        # mask = decode_jpeg(mask)

        # mask = tf.convert_to_tensor(mask)
        # mask = tf.image.convert_image_dtype(mask, tf.float32)

        return ID, img[:, :, :3], mask, image_size, organ, org_id, sex_id, age

    def resize_image(self, img):
        new_img = tf.image.resize(img, (self.TARGET_WIDTH, self.TARGET_WIDTH),
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return new_img

    def resize_mask(self, mask):
        mask = tf.expand_dims(mask, 2)
        new_mask = tf.image.resize(mask, (self.TARGET_WIDTH, self.TARGET_WIDTH),
                                   method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        new_mask = tf.cast(new_mask, dtype=tf.float32)
        new_mask = tf.transpose(new_mask, perm=(2, 1, 0))
        return new_mask
    
    def normalize_image(self, image):
        image = tf.cast(image, dtype=tf.float32)
        norm_image = image / 255.0
        norm_image = tf.transpose(norm_image, perm=(2, 1 , 0))
        return norm_image

    def sex_to_num(self):
        label_encoder = preprocessing.LabelEncoder()
        self.df['sex_id'] = label_encoder.fit_transform(self.df['sex'])
        return self.df['sex_id'].values

    def org_name_to_num(self):
        label_encoder = preprocessing.LabelEncoder()
        self.df['org_id'] = label_encoder.fit_transform(self.df['organ'])
        return self.df['org_id'].values

    def metadataset_from_files(self, frame):
        values = {
            'id': frame.id.values.astype(np.int32),
            'fname': frame['image'].values,
            'mname': frame['rle'],
            'width': tf.convert_to_tensor(frame['img_width'].values),
            'height': tf.convert_to_tensor(frame['img_height'].values),
            'organ': frame['organ'].values,
            'org_id': tf.convert_to_tensor(self.org_name_to_num()),
            'sex_id': tf.convert_to_tensor(self.sex_to_num()),
            'age': tf.convert_to_tensor(frame['age'].values.astype(np.int32))
        }
        metadataset = tf.data.Dataset.from_tensor_slices(values)

        return metadataset

    def display_preview(self):
        filenames_to_process = self.image_files.copy()
        print("TFRecords output directory: {}".format(self.OUTPUT_DIR))
        print("Image resizing target size: {}px".format(self.TARGET_WIDTH))
        print("Number of images: {}".format(len(filenames_to_process)))
        NB_SHARDS = -(-len(filenames_to_process) // self.IMAGES_PER_SHARD)
        print(
            f"Output sharded into {NB_SHARDS} files with {self.IMAGES_PER_SHARD} images per file, {len(filenames_to_process) - (NB_SHARDS - 1) * self.IMAGES_PER_SHARD} images in last file")

    def compute_id_bytestring(self, s):
        computed_id = s
        return str(computed_id).encode('utf-8')

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(
            value=value))  # WARNING: this expects a list of byte strings, not a list of bytes!

    def _int_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def to_tfrecord(self, tfrec_filewriter, ids, img_bytes, mask_bytes, widths, heights, organs, org_ids, sex_ids,
                    ages):
        # mask_bytes = str(mask_bytes).encode('utf-8')
        # print(mask_bytes)
        feature = {
            "image/encoded": self._bytes_feature([img_bytes]),  # Compressed image bytes
            "image/source_id": self._int_feature([ids]),  # ID
            "image/width": self._int_feature([widths]),  # image width
            "image/height": self._int_feature([heights]),  # image height
            "image/mask": self._bytes_feature([mask_bytes]),  # mask rle
            "image/organ": self._bytes_feature([self.compute_id_bytestring(organs)]),  # organ name
            "image/org_id": self._int_feature([org_ids]),  # organ id
            "image/sex_id": self._int_feature([sex_ids]),  # sex id
            "image/age": self._int_feature([ages])  # subject age
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def recompress_jpeg(self, image):
        image = tf.image.convert_image_dtype(image, tf.uint8)
        return tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)

    def write_dataset(self):
        self.df['rle'] = self.df.apply(self.encode_rle, axis=1)
        mask_files = glob.glob("./train_masks/*")
        filenames_to_process = self.image_files.copy()
        NB_SHARDS = -(-len(filenames_to_process) // self.IMAGES_PER_SHARD)

        self.df['image'] = self.image_files
        # self.df['mask'] = mask_files
        self.display_preview()

        metadataset = self.metadataset_from_files(frame=self.df.sample(frac=1.0).reset_index(drop=True))
        dataset = metadataset.map(self.read_and_decode, num_parallel_calls=AUTO)

        dataset = dataset.map(lambda ID, img, mask, size, organ, org_id, sex_id, age:
                              (ID, self.recompress_jpeg(img), mask, size, organ, org_id, sex_id, age),
                              num_parallel_calls=AUTO)

        dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(self.IMAGES_PER_SHARD))
        TEST_SHARDS = [4, 8, 10, 15, 17, 20, 27, 31]  # these files will be labeled "test" other files "train"

        if os.path.isdir(self.OUTPUT_DIR) and len(os.listdir(self.OUTPUT_DIR)) > 0:
            print(
                "ERROR: the output directory exists and is not empty. Aborting. Please empty the output directory manually before proceeding.")
        else:
            my_dir = True
            print("Making Directory...\n")
            if my_dir:
                os.mkdir(self.OUTPUT_DIR)
                print("Writing TFRecords...")
                for shard, (ID, img, mask, size, organ, org_id, sex_id, age) in enumerate(dataset):
                    shard_size = img.numpy().shape[0]
                    filename = self.OUTPUT_NAME_PREFIX + "w{}px_{:03d}_of_{:03d}-{:03d}.{}.tfrec".format(
                        self.TARGET_WIDTH, shard + 1, NB_SHARDS, shard_size,
                        'test' if shard in TEST_SHARDS else 'train')
                    with tf.io.TFRecordWriter(os.path.join(self.OUTPUT_DIR, filename)) as file:
                        for i in range(shard_size):
                            binary_id = ID[i].numpy()
                            binary_image = img[i].numpy()
                            binary_mask = mask[i].numpy()
                            binary_width = size[i].numpy()[0]
                            binary_height = size[i].numpy()[1]
                            binary_organ = self.compute_id_bytestring(organ[i].numpy().decode('utf-8'))
                            binary_org_id = org_id[i].numpy()
                            binary_sex_id = sex_id[i].numpy()
                            binary_age = age[i].numpy()

                            example = example = self.to_tfrecord(file, binary_id, binary_image, binary_mask,
                                                                 binary_width, binary_height,
                                                                 binary_organ, binary_org_id, binary_sex_id, binary_age)
                            file.write(example.SerializeToString())
                    print("Wrote file {} containing {} records".format(filename, shard_size))

    def read_tfrecord(self, example):
        feature = {
            "image/source_id": tf.io.FixedLenFeature([], tf.int64),
            "image/encoded": tf.io.FixedLenFeature([], tf.string),
            "image/width": tf.io.FixedLenFeature([], tf.int64),
            "image/height": tf.io.FixedLenFeature([], tf.int64),
            "image/mask": tf.io.FixedLenFeature([], tf.string),
            "image/organ": tf.io.FixedLenFeature([], tf.string),
            "image/org_id": tf.io.FixedLenFeature([], tf.int64),
            "image/sex_id": tf.io.FixedLenFeature([], tf.int64),
            "image/age": tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, feature)
        source_id = example["image/source_id"]
        image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
        width = example["image/width"]
        height = example["image/height"]
        mask = example['image/mask']
        organ = example["image/organ"]
        org_id = example["image/org_id"]
        sex_id = example['image/sex_id']
        age = example["image/age"]

        return source_id, image, mask, width, height, organ, org_id, sex_id, age

    def load_tfrecord_dataset(self, filenames):
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False

        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
        dataset = dataset.with_options(ignore_order)
        dataset = dataset.map(self.read_tfrecord, num_parallel_calls=AUTO)
        return dataset

    def get_train_dataset(self):
        filenames = tf.io.gfile.glob(os.path.join(self.OUTPUT_DIR, "*train.tfrec"))
        dataset = self.load_tfrecord_dataset(filenames)
        print("Dataset Loaded.")

        # Converting rles to masks
        dataset = dataset.map(lambda ID, img, rle, width, height, organ, org_id, sex_id, age:
                              (ID, img, rle, self.rle2mask(rle, shape=(3000, 3000)), width, height, organ, org_id,
                               sex_id, age), num_parallel_calls=AUTO)

        # resize images and masks
        dataset = dataset.map(lambda source_ids, images, rles, masks, widths, heights, organs, org_ids, sex_ids, ages:
                              (source_ids, self.normalize_image(self.resize_image(images)), rles,
                               self.resize_mask(masks), widths, heights, organs, org_ids, sex_ids, ages),
                              num_parallel_calls=AUTO)
        dataset = dataset.map(lambda source_ids, images, rles, masks, widths, heights, organs, org_ids, sex_ids, ages:
                              (images, org_ids, sex_ids, ages, masks), num_parallel_calls=AUTO)
        dataset_iterator = iter(dataset.apply(tf.data.experimental.dense_to_ragged_batch(self.IMAGES_PER_SHARD)))
        
        return dataset_iterator

    def get_test_dataset(self):
        filenames = tf.io.gfile.glob(os.path.join(self.OUTPUT_DIR, "*test.tfrec"))
        dataset = self.load_tfrecord_dataset(filenames)
        print("Dataset Loaded.")

        # Converting rles to masks
        dataset = dataset.map(lambda ID, img, rle, width, height, organ, org_id, sex_id, age:
                              (ID, img, rle, self.rle2mask(rle, shape=(3000, 3000)), width, height, organ, org_id,
                               sex_id, age), num_parallel_calls=AUTO)

        # resize images and masks
        dataset = dataset.map(lambda source_ids, images, rles, masks, widths, heights, organs, org_ids, sex_ids, ages:
                              (source_ids, self.normalize_image(self.resize_image(images)), rles,
                               self.resize_mask(masks), widths, heights, organs, org_ids, sex_ids, ages),
                              num_parallel_calls=AUTO)
        dataset = dataset.map(lambda source_ids, images, rles, masks, widths, heights, organs, org_ids, sex_ids, ages:
                              (images, org_ids, sex_ids, ages, masks), num_parallel_calls=AUTO)
        dataset_iterator = iter(dataset.apply(tf.data.experimental.dense_to_ragged_batch(self.IMAGES_PER_SHARD)))
        return dataset_iterator

    def get_combined_dataset(self):
        filenames = tf.io.gfile.glob(os.path.join(self.OUTPUT_DIR, "*.tfrec"))
        dataset = self.load_tfrecord_dataset(filenames)
        print("Dataset Loaded.")

        # Converting rles to masks
        dataset = dataset.map(lambda ID, img, rle, width, height, organ, org_id, sex_id, age:
                              (ID, img, rle, self.rle2mask(rle, shape=(3000, 3000)), width, height, organ, org_id,
                               sex_id, age), num_parallel_calls=AUTO)

        # resize images and masks
        dataset = dataset.map(lambda source_ids, images, rles, masks, widths, heights, organs, org_ids, sex_ids, ages:
                              (source_ids, self.normalize_image(self.resize_image(images)), rles,
                               self.resize_mask(masks), widths, heights, organs, org_ids, sex_ids, ages),
                              num_parallel_calls=AUTO)
        dataset = dataset.map(lambda source_ids, images, rles, masks, widths, heights, organs, org_ids, sex_ids, ages:
                              (images, org_ids, sex_ids, ages, masks), num_parallel_calls=AUTO)
        dataset_iterator = iter(dataset.apply(tf.data.experimental.dense_to_ragged_batch(self.IMAGES_PER_SHARD)))
        
        return dataset_iterator
    
    def get_train_kfold(self):
        filenames = tf.io.gfile.glob(os.path.join(self.OUTPUT_DIR, "*train.tfrec"))
        
        fold_iters = []
        for tr_idx, ts_idx in self.kfold.split(filenames):
            fold_train = []
            fold_test = []
            #print(tr_idx)
            for idx in tr_idx:
                fold_train.append(filenames[idx])
            
            for t_idx in ts_idx:
                fold_test.append(filenames[t_idx])

            assert any(item in fold_train for item in fold_test) == False, "Overlap in train/test exists"
        
            dataset_train = self.load_tfrecord_dataset(fold_train)
            dataset_test = self.load_tfrecord_dataset(fold_test)
            
            # Converting rles to masks
            dataset_train = dataset_train.map(lambda ID, img, rle, width, height, organ, org_id, sex_id, age:
                                              (ID, img, rle, self.rle2mask(rle, shape=(3000, 3000)), width, height, organ, org_id, sex_id, age), num_parallel_calls=AUTO)
            dataset_test = dataset_test.map(lambda ID, img, rle, width, height, organ, org_id, sex_id, age:
                                            (ID, img, rle, self.rle2mask(rle, shape=(3000, 3000)), width, height, organ, org_id, sex_id, age), num_parallel_calls=AUTO)
            # resize images and masks
            dataset_train = dataset_train.map(lambda source_ids, images, rles, masks, widths, heights, organs, org_ids, sex_ids, ages:
                                              (source_ids, self.normalize_image(self.resize_image(images)), rles, 
                                               self.resize_mask(masks), widths, heights, organs, org_ids, sex_ids, ages),
                                              num_parallel_calls=AUTO)
            dataset_test = dataset_test.map(lambda source_ids, images, rles, masks, widths, heights, organs, org_ids, sex_ids, ages:
                                            (source_ids, self.normalize_image(self.resize_image(images)), rles, 
                                             self.resize_mask(masks), widths, heights, organs, org_ids, sex_ids, ages),
                                            num_parallel_calls=AUTO)
            
            dataset_train = dataset_train.map(lambda source_ids, images, rles, masks, widths, heights, organs, org_ids, sex_ids, ages:
                                              (images, org_ids, sex_ids, ages, masks), num_parallel_calls=AUTO)
            dataset_test = dataset_test.map(lambda source_ids, images, rles, masks, widths, heights, organs, org_ids, sex_ids, ages:
                                            (images, org_ids, sex_ids, ages, masks), num_parallel_calls=AUTO)
                
            dataset_fold_train = iter(dataset_train.apply(tf.data.experimental.dense_to_ragged_batch(self.IMAGES_PER_SHARD)))
            dataset_fold_test = iter(dataset_test.apply(tf.data.experimental.dense_to_ragged_batch(self.IMAGES_PER_SHARD)))
            fold_tup = (dataset_fold_train, dataset_fold_test)
            fold_iters.append(fold_tup)
        
        print("Dataset Loaded\n.")
        print(f"Dataset is split into {len(fold_iters)} folds")
        
        return fold_iters


def load_data():
    my_dataset = tissue_dataset(train_csv="./train.csv", images_per_shard=10)

    #my_dataset.show_example()
    # my_dataset.save_masks()
    # my_dataset.write_dataset()

    #train_dataset_iterator, test_dataset_iterator = my_dataset.get_train_kfold()
    
    fold_iters = my_dataset.get_train_kfold()
    (train_dataset_iterator, test_dataset_iterator) = fold_iters[0]
    

    train_images, train_org_ids, train_sex_ids, train_ages, train_masks = next(train_dataset_iterator)
    test_images, test_org_ids, test_sex_ids, test_ages, test_masks = next(test_dataset_iterator)

    print(train_org_ids[0])
    print(test_org_ids[0])

    print(train_images[0].shape)
    print(test_images[0].shape)

    print(train_masks[0].shape)

    print(test_masks[0].dtype)
    print(test_images[0].dtype)

    # train_images = tf.concat([train_images, test_images], axis=0)

    plt.imshow(rechannel(train_images[0].numpy()))
    plt.imshow(rechannel(train_masks[0].numpy()), cmap='coolwarm', alpha=0.5)

    plt.figure()

    plt.imshow(rechannel(test_images[0].numpy()))
    plt.imshow(rechannel(test_masks[0].numpy()), cmap='coolwarm', alpha=0.5)

    
'''
    rle_check = my_dataset.mask2rle(my_dataset.get_orig_mask(train_masks[0]).numpy())
    rle_true = train_rles[0].numpy()
    
    print(train_source_ids[0])
    
    print(rle_check)
    print("\n----------------------------------\n")
    print(rle_true)
'''
    
    


if __name__ == "__main__":
    load_data()
