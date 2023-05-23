#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 10:36:36 2022

@author: hal
"""
import tensorflow as tf
import keras
from keras.losses import CategoricalCrossentropy
from model_cast import Flatten
import cv2

from ETL_improved import tissue_dataset
from modelling import image_model
from modelling import model_config 

import datetime


class DiceLoss(keras.layers.Layer):
    def __init__(self, n_classes):
        super().__init__()
        self.trainable = False
        self.n_classes = n_classes
        
    def one_hot_enc(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(tf.expand_dims(temp_prob, axis=1))
        
        output_tensor = tf.concat(values=tensor_list, axis=1)
        
        return tf.cast(output_tensor, dtype=tf.float32)
    
    def dice_loss(self, score, target):
        target = tf.cast(target, dtype=tf.float32)
        smooth = 1e-5
        intersect = tf.reduce_sum(score * target)
        y_sum = tf.reduce_sum(target * target)
        z_sum = tf.reduce_sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    
    def call(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = tf.math.softmax(logits=inputs, axis=1)
        target = self.one_hot_enc(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.shape == target.shape, f"Prediction shape: {inputs.shape} and target shape: {target.shape} do not match"
        
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self.dice_loss(score=inputs[:, i], target=target[:, i])
            loss += dice * weight[i]
        
        return loss / self.n_classes
    
class MultiClassDiceCE(keras.layers.Layer):
    def __init__(self, num_classes=3, CE_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.trainable = False
        self.CE_loss = CategoricalCrossentropy(from_logits=True)
        self.dice_loss = DiceLoss(num_classes)
        self.CE_weight = CE_weight
        self.dice_weight = dice_weight
        self.flatten = Flatten()
        
    def show_dice(self, inputs, targets, softmax=True):
        dice = 1.0 - self.dice_loss(inputs, targets, softmax=softmax)
        return dice
    
    def call(self, inputs, targets, softmax=True):
        dice = self.dice_loss(inputs, targets, softmax=softmax)
        CE = self.CE_loss(y_true=targets, y_pred=inputs)
        dice_CE_loss = (self.dice_weight * dice) + (self.CE_weight * CE)
        
        return dice_CE_loss
    
class TissueTrainer:
    def __init__(self, model, optimizer, model_config, training_config, seed=42, training=True):
        self.training = training
        self.training_config = training_config
        self.model_config = model_config()
        self.model_config = self.model_config.get_config()
        
        self.seed = seed
        self.epochs = training_config['epochs']
        self.kfolds = training_config['kfolds']
        
        self.dataset = tissue_dataset(train_csv="./train.csv", images_per_shard=10, target_width=224, k_folds=self.kfolds)
        self.model = model(seed=self.seed, config=self.model_config)
        self.optimizer = optimizer
            
        self.flatten = Flatten()
        
        self.dice_loss = MultiClassDiceCE()
        
    def dice_coeff(self, y_true, y_pred):
        smooth = 1e-5
        y_true_f = self.flatten(y_true)
        y_pred_f = self.flatten(y_pred)
        intersection = tf.reduce_sum(y_true_f, y_pred_f)
        
        return (2 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    def dice_on_batch(self, masks, pred):
        dices = []
        for i in range(pred.shape[0]):
            pred_tmp = pred[i][0]
            mask_tmp = masks[i]
            
            pred_tmp = tf.where(tf.greater_equal(pred_tmp, tf.constant(0.5)), tf.constant(1), tf.constant(0))            
            mask_tmp = tf.where(tf.greater(mask_tmp, tf.constant(0)), tf.constant(1), tf.constant(0))
            
            dices.append(self.dice_coeff(mask_tmp, pred_tmp))
        
        return tf.reduce_mean(dices)
    
    def save_on_batch(self, images1, masks, pred, names, vis_path):
        for i in range(pred.shape[0]):
            pred_tmp = pred[i][0]
            mask_tmp = masks[i]
            
            pred_tmp = tf.where(tf.greater_equal(pred_tmp, tf.constant(0.5)), tf.constant(255), tf.constant(0))
            mask_tmp = tf.where(tf.greater(mask_tmp, tf.constant(0)), tf.constant(255), tf.constant(0))
            
            cv2.imwrite(vis_path+names[i][:-4]+"_pred.jpg", pred_tmp)
            cv2.imwrite(vis_path+names[i][:-4]+"_gt.jpg", mask_tmp)
            
    def grad(self, features, targets, train_iter):
        with tf.GradientTape() as tape:
            preds, targets = self.model(features, targets)
            loss_value = self.dice_loss(preds, targets)
            if train_iter:
                return loss_value, tape.gradient(loss_value, self.model.trainable_variables)
            else:
                return loss_value
    
    def train_per_epoch(self, train_iterator, epoch, train_iter=True):
        epoch_loss_avg = keras.metrics.Mean()
        epoch_acc = keras.metrics.CategoricalAccuracy()
        
        for batch in train_iterator:
            images = batch[0]
            masks = batch[-1]
            loss_value, grads = self.grad(images, masks, train_iter)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            epoch_loss_avg.update_state(loss_value)
            predictions, truths = self.model(images, masks)
            epoch_acc.update_state(truths, predictions)
        
        avg_loss = epoch_loss_avg.result()
        avg_acc = epoch_acc.result()
            
        print(f"Epoch {epoch} -- Average Training Loss: {avg_loss}, Average Training Accuracy: {avg_acc}")
            
        return avg_loss, avg_acc
    
    def val_per_epoch(self, val_iterator, epoch, train_iter=False):
        epoch_loss_avg = keras.metrics.Mean()
        epoch_acc = keras.metrics.CategoricalAccuracy()
        
        for batch in val_iterator:
            images = batch[0]
            masks = batch[-1]
            loss_value = self.grad(images, masks, train_iter)
            
            epoch_loss_avg.update_state(loss_value)
            predictions, truths = self.model(images, masks)
            epoch_acc.update_state(truths, predictions)
            
        avg_loss = epoch_loss_avg.result()
        avg_acc = epoch_acc.result()
        
        print(f"Epoch {epoch} -- Average Validation Loss: {avg_loss}, Average Validation Accuracy: {avg_acc}")
        
        return avg_loss, avg_acc
        
    
    def training_loop(self, data_iterator, fold_num):
        train_iterator = data_iterator[0]
        val_iterator = data_iterator[1]
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = './tissue_logs/gradient_tape/' + current_time + '/train'
        val_log_dir = './tissue_logs/gradient_tape/' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        
        chkpt = tf.train.Checkpoint(epoch=tf.Variable(0), optimizer=self.optimizer, model=self.model, iterator=data_iterator)
        manager = tf.train.CheckpointManager(checkpoint=chkpt, directory=f"./tissue_chkpts/{fold_num}", max_to_keep=2)
        chkpt.restore(manager.latest_checkpoint)
        
        if manager.latest_checkpoint:
            print(f"Restored from {manager.latest_checkpoint}")
        
        else:
            print("Starting from scratch")
        
        
        for epoch in range(self.epochs):
            train_epoch_loss, train_epoch_acc = self.train_per_epoch(train_iterator, epoch)
            train_losses.append(train_epoch_loss)
            train_accs.append(train_epoch_acc)
            
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", train_epoch_loss, step=epoch)
                tf.summary.scalar("accuracy", train_epoch_acc, step=epoch)
            
            val_epoch_loss , val_epoch_acc  = self.val_per_epoch(val_iterator, epoch)
            val_losses.append(val_epoch_loss)
            val_accs.append(val_epoch_acc)
            
            with val_summary_writer.as_default():
                tf.summary.scalar("loss", val_epoch_loss, step=epoch)
                tf.summary.scalar("accuracy", val_epoch_acc, step=epoch)
            
            chkpt.epoch.assign(epoch)
            if epoch % 5 == 0:
                save_path = manager.save()
                print(f"Saved checkpoint for epoch {epoch} in: {save_path}")
                
        return tf.convert_to_tensor(train_losses), tf.convert_to_tensor(train_accs), tf.convert_to_tensor(val_losses), tf.convert_to_tensor(val_accs)
    
    def train_kfold(self):
        print(f"\n--------------Beginning K-Fold Training of {self.kfolds} folds---------------------------\n")
        k_fold_iterators = self.dataset.get_train_kfold()
        num_fold = 1
        
        for fold_data in k_fold_iterators:
            print(f"Beginning fold {num_fold} training...\n")
            train_losses, train_accs, val_losses, val_accs = self.training_loop(fold_data, num_fold)
            avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc = tf.reduce_mean(train_losses), tf.reduce_mean(train_accs), tf.reduce_mean(val_losses), tf.reduce_mean(val_accs)
            
            print(f"...Fold {num_fold} metrics...\nTraining Loss: {avg_train_loss}; Training Accuracy: {avg_train_acc}\nValidation Loss: {avg_val_loss}; Validation Accuracy: {avg_val_acc}")
            
        self.model.save_weights('tissue_weights.h5')
        print("\n\n---------------Finished training all folds and saved model-----------------------")
        
    def get_model_arch(self):
        input_shape = (10, 3, 224, 224)
        self.model.build(input_shape)
        keras.utils.plot_model(self.model, "tissue_model.png", show_shapes=True)
        

def train_model():
    model = image_model
    optimizer = keras.optimizers.Adam()
    
    training_config = {"epochs": 20,
                       "kfolds": 7
                       }
    
    model_trainer = TissueTrainer(model, optimizer, model_config, training_config)
    print("Trainer Instantiated--------- .....Beginning Training.....----------------\n")
    model_trainer.train_kfold()
    
if __name__ == "__main__":
    train_model()
    
    
            
            
        
        
