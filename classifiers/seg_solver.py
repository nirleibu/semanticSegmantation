from random import shuffle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import SGD, Adam


class Solver(object):

    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={}):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim

        weight = torch.ones(22)  # todo - maybe change to 23
        weight[0] = 0
        self.loss = nn.NLLLoss2d(weight)

        self._reset_histories()

    def loss_func(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        print 'START TRAIN.'
        ############################################################################
        # TODO:                                                                    #
        # Write your own personal training method for our solver. In Each epoch    #
        # iter_per_epoch shuffled training batches are processed. The loss for     #
        # each batch is stored in self.train_loss_history. Every log_nth iteration #
        # the loss is logged. After one epoch the training accuracy of the last    #
        # mini batch is logged and stored in self.train_acc_history.               #
        # We validate at the end of each epoch, log the result and store the       #
        # accuracy of the entire validation set in self.val_acc_history.           #
        #
        # Your logging should like something like:                                 #
        #   ...                                                                    #
        #   [Iteration 700/4800] TRAIN loss: 1.452                                 #
        #   [Iteration 800/4800] TRAIN loss: 1.409                                 #
        #   [Iteration 900/4800] TRAIN loss: 1.374                                 #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                                #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                                #
        #   ...                                                                    #
        ############################################################################

        model.train()

        optimizer = SGD(model.parameters(), 1e-4, .9, 2e-5)
        # optimizer = Adam(model.parameters())

        for epoch in range(num_epochs):  # loop over the dataset multiple times

            print 'epoch: %d' % epoch

            for step, (images, labels) in enumerate(train_loader):

                print 'step: %d' % step

                inputs = Variable(images)
                targets = Variable(labels)
                outputs = model(inputs)

                optimizer.zero_grad()
                train_loss = self.loss_func(outputs, targets[:, 0])
                train_loss.backward()
                optimizer.step()

                self.train_loss_history.append(train_loss.data[0])

                # print statistics
                if step % log_nth == log_nth - 1:
                    print('[epoch: %d, iteration: %d] loss: %f' %
                          (epoch + 1, step + 1, train_loss.data[0]))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        print 'FINISH.'
