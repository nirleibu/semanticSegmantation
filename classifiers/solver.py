from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func()

        self._reset_histories()

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

        for epoch in range(num_epochs):  # loop over the dataset multiple times

            for i, data in enumerate(train_loader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                train_inputs, train_labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optim.zero_grad()

                # forward + backward + optimize
                train_outputs = model(train_inputs)
                train_loss = self.loss_func(train_outputs, train_labels)
                train_loss.backward()
                optim.step()

                self.train_loss_history.append(train_loss.data[0])

                # print statistics
                if i % log_nth == log_nth - 1:
                    print('[epoch: %d, iteration: %d] loss: %f' %
                          (epoch + 1, i + 1, train_loss.data[0]))

            # check training accuracy on last minibatch
            _, predicted = torch.max(train_outputs.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum()
            print('Training accuracy epoch %d: %d %%' % (epoch, 100 * correct / total))
            self.train_acc_history.append(100 * correct / total)

            # check validation accuracy on entire set
            correct = 0
            total = 0
            for val_data in val_loader:
                val_images, val_labels = val_data
                val_outputs = model(Variable(val_images))
                _, predicted = torch.max(val_outputs.data, 1)
                total += val_labels.size(0)
                correct += (predicted == val_labels).sum()
            print('Validation accuracy epoch %d: %d %%' % (epoch, 100 * correct / total))
            self.val_acc_history.append(100 * correct / total)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        print 'FINISH.'
