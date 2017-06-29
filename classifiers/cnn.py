import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init
import numpy as np


class ThreeLayerCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 stride=1, weight_scale=0.001, pool=2, hidden_dim=100,
                 num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - filter_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden layer-
        - num_classes: Number of scores to produce from the final affine layer.
        - stride: The size of the window to take a max over.
        - weight_scale: Scale for the convolution weights initialization-
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ThreeLayerCNN, self).__init__()
        channels, height, width = input_dim

        ############################################################################
        # TODO: Initialize the necessary layers to resemble the ThreeLayerCNN      #
        # architecture  from the class docstring. In- and output features should   #
        # not be hard coded which demands some calculations especially for the     #
        # input of the first fully convolutional layer. The convolution should use #
        # "same" padding which can be derived from the kernel size and its weights #
        # should be scaled. Layers should have a bias if possible.                 #
        ############################################################################

        p = (filter_size - 1) / 2 if stride == 1 else 0

        conv_out_width = (width - filter_size + 2 * p) / stride + 1
        conv_out_height = (height - filter_size + 2 * p) / stride + 1

        pool_out_width = (conv_out_width - pool) / pool + 1
        pool_out_height = (conv_out_height - pool) / pool + 1

        self.pool = pool
        self.dropout = dropout

        self.conv = nn.Conv2d(in_channels=channels, out_channels=num_filters, kernel_size=filter_size,
                              padding=p,
                              stride=stride,
                              bias=True)
        weight_init.xavier_normal(self.conv.weight, gain=np.sqrt(2))
        weight_init.constant(self.conv.bias, weight_scale)

        lin_in = pool_out_width * pool_out_height * num_filters

        self.fc1 = nn.Linear(in_features=lin_in, out_features=hidden_dim, bias=True)
        weight_init.xavier_normal(self.fc1.weight, gain=np.sqrt(2))
        weight_init.constant(self.fc1.bias, weight_scale)

        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=num_classes, bias=True)
        weight_init.xavier_normal(self.fc2.weight, gain=np.sqrt(2))
        weight_init.constant(self.fc2.bias, weight_scale)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        ############################################################################
        # TODO: Chain our previously initialized convolutional neural network      #
        # layers to resemble the architecture drafted in the class docstring.      #
        # Have a look at the Variable.view function to make the transition from    #
        # convolutional to fully connected layers.                                 #
        ############################################################################

        x = self.conv(x)
        x = F.relu(F.max_pool2d(x, kernel_size=self.pool))
        (_, C, H, W) = x.data.size()
        x = x.view(-1, C * H * W)
        x = F.relu(F.dropout(self.fc1(x), p=self.dropout))
        x = self.fc2(x)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print 'Saving model... %s' % path
        torch.save(self, path)
