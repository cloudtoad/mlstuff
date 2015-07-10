import apollo
from apollo import layers
import numpy as np

net = apollo.Net()
for i in range(1000000):

## Randomly generate two values.  
    left = np.array(np.round(np.random.random(),3)).reshape((1,1,1,1))
    right = np.array(np.round(np.random.random(),3)).reshape((1,1,1,1))

## These 'layers' are where data is input into the CNN
    net.forward_layer(layers.NumpyData(name='left', data=left))
    net.forward_layer(layers.NumpyData(name='right', data=right))

## Submit the label
    net.forward_layer(layers.NumpyData(name='label', data=left+right))

## Concatenate the output of the "left" and "right" layers.
    net.forward_layer(layers.Concat(name="concat", bottoms=['left', 'right'], axis=1))

## The actual convolution.
    net.forward_layer(layers.Convolution(name='conv', kernel_size=1, bottoms=['concat'], num_output=1))

## Loss layer.  Takes output of convolution layer and the label and produces a loss.
    loss = net.forward_layer(layers.EuclideanLoss(name='loss', bottoms=['conv', 'label']))

## Back propogate and update hidden variables.
    net.backward()
    net.update(lr=0.2)

## Print one result every ten thousand transactions.
    if i % 10000 == 0:
	print i, left, right, net.tops['conv'].data.flatten()[0], left+right, loss




