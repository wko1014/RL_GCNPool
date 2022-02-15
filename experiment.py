from utils import call_dataset

import network
import tensorflow as tf
import numpy as np
import utils

class experiment():
    def __init__(self, fold_idx):
        self.fold_idx = fold_idx

        # Learning schedules
        self.initial_learning_rate = 1e-3
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_learning_rate, decay_steps=10000, decay_rate=.96, staircase=False)
        self.num_epochs = 50
        self.num_batches = 5
        self.optimizer = tf.keras.optimizers.Nadam(learning_rate=self.initial_learning_rate)
        self.pre_training = 10

    def training(self):
        print(f'START TRAINING FOLD {self.fold_idx}')
        test_auc = 0 ##

        # Load dataset
        D_train, D_valid, D_test = call_dataset(self.fold_idx, one_hot=True)
        X_train, Y_train = D_train
        X_valid, Y_valid = D_valid
        X_test, Y_test = D_test

        # Call model
        fMRINet = network.GCIR3_v1()

        # Optimization
        optimizer = self.optimizer
        num_batch_iter = int(X_train.shape[0]/self.num_batches)

        for epoch in range(self.num_epochs):
            if epoch < self.pre_training:
                RL = False
            else: RL = True
            loss_per_epoch = 0
            # Randomize the training dataset
            rand_idx = np.random.permutation(X_train.shape[0])
            X_train, Y_train = X_train[rand_idx, :, :, :], Y_train[rand_idx, :]

            for batch in range(num_batch_iter):
                # Sample minibatch
                x = X_train[batch * self.num_batches:(batch + 1) * self.num_batches, :, :, :] # [5, 114, 130, 1]
                y = Y_train[batch * self.num_batches:(batch + 1) * self.num_batches, :]

                # # Sample next minibatch for the agent training
                x_next = X_train[(batch + 1) * self.num_batches:(batch + 2) * self.num_batches, :, :, :]
                y_next = Y_train[(batch + 1) * self.num_batches:(batch + 2) * self.num_batches, :]

                # Estimate loss
                loss, grads = utils.gradient(model=fMRINet, inputs=x, labels=y, rl=RL)

                # # Estimate loss of next minibatch for the agent training
                _, grads_agent = utils.gradient(model=fMRINet, inputs=x_next, labels=y_next, rl=RL)

                # Update the parameters
                optimizer.apply_gradients(zip(grads, fMRINet.trainable_variables))

                loss_per_epoch += np.mean(loss)

            Y_valid_hat = fMRINet.call(X_valid, rl=RL)
            _auc, _acc, _sen, _spec = utils.evaluate(pred=Y_valid_hat, lab=Y_valid, one_hot=True)

            Y_test_hat = fMRINet.call(X_test, rl=RL)
            auc, acc, sen, spec = utils.evaluate(pred=Y_test_hat, lab=Y_test, one_hot=True)

            print(f'Valid AUC: {_auc:.4f}, ACC: {_acc:.4f}, SEN: {_sen:.4f}, SPEC: {_spec:.4f}')
            print(f'Testing AUC: {auc:.4f}, ACC: {acc:.4f}, SEN: {sen:.4f}, SPEC: {spec:.4f}')

            
fold = 1 # fold to conduct
main = experiment(fold_idx=fold)
main.training()
