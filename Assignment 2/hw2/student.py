#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

"""
Question: Briefly describe how your program works, and explain any design and 
    training decisions you made along the way.

How it works:
    This program is mainly divided into three components, namely preprocessing, 
    neural network and loss function.

    In brief, the workflow of this program is to pass the processed input into 
    the network, then train rating and category task through its sub-networks 
    rateNet and cateNet (with attention involved in cateNet) respectively, and return 
    the results to the network. Then the loss function is used to calculate the 
    loss. Finally, optimiser is used to make different optimizations for 
    different neural netwokrs.
    
Design and training decisions:
    
    Some Lemmatizer, Stemmer and N-grams methods were used, but the accuracy was
    found to be reduced in multiple testings. Therefore, we decided to abandon 
    these methods.

    The network is designed to have 2 neural networks, one is learning rate 
    recognition, other is learning category recognition. The reason for 
    separating the networks is we found these two tasks appear to prefer 
    different learning during training. Therefore, an optimizer which applies 
    different learning rate to the network is decided to use during training. 
    On the other hand, dropout is applied in LSTM and fully connect layer to 
    prevent overfitting. For improving the training speed and accuracy, 
    pack_padding_sequence is used in training.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
import torch.nn.functional as F
import re
# import numpy as np
# import sklearn

from config import device

# parameters
EMBED_SIZE = 200
HIDDEN_SIZE = 128
DROPOUT = 0.5
NUM_LAYERS = 2
BIDIRECTIONAL = True


################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = re.split(" |!|\?|\.|\n|'|\(|\)", sample) # Split the sample via the regular expression.
    return list(filter(lambda item: item, processed))


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """

    return sample


def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch


stopWords = {',','.','(',')','$','1','2','3','4','5','6','7','8','9','0','-'}
wordVectors = GloVe(name='6B', dim=EMBED_SIZE) # EMBED_SIZE = 200


################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    Args:
        ratingOutput: the prediction of rating via the network.
        categoryOutput: the prediction of category via the network.
    Returns:
        ratingOutput: LongTensor of indices for the maximum values of the ratingOutput across the dim = 1.
        categoryOutput: LongTensor of indices for the maximum values of the categoryOutput across the dim = 1.
    """
    # Convert to integer values: The indices of the maximum values of the ratingOutput and categoryOutput across the dim = 1.
    # Convert the type to LongTensor.
    ratingOutput = ratingOutput.argmax(dim=1).long() # shape: [batch_size].
    categoryOutput = categoryOutput.argmax(dim=1).long() # shape: [batch_size].

    return ratingOutput, categoryOutput


################################################################################
###################### The following determines the model ######################
################################################################################
class rateNet(tnn.Module):

    def __init__(self, embeded_size, hidden, num_layer, dropout, bidirectional):
        """Declare the attributes of class rateNet.
        Args:
            embeded_size: the input_size of torch.nn.LSTM
            hidden: the hidden_size of torch.nn.LSTM
            num_layer: the num_layers of torch.nn.LSTM
            dropout: the dropout rate torch.nn.LSTM
            bidirectional: True or False to declare bidirectional or unidirectional of torch.nn.LSTM
        """
        super(rateNet, self).__init__()
        self.lstm = tnn.LSTM(embeded_size, hidden, num_layer, dropout=dropout, bidirectional=bidirectional)
        self.rateLinear = tnn.Linear(hidden * (2 if bidirectional else 1), 2) # if bidirectional == True: hidden * 2, else: hidden * 1
        self.dropout = tnn.Dropout(dropout)

    def forward(self, input):
        """Predict a category based on the input via cateNet.
        Args:
            input: shape => [batch_size, sentence_length, dim] => [32, flex, 200].
                Embedded business reviews after packed.
        Returns:
            ratingOutput: shape => [batch_size, 2].
                Predicted rating of the reviews (positive or negative).
        """
        output, (hidden, cell) = self.lstm(input)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1) # hidden: [batch_size, 200]
        hidden = self.dropout(hidden)
        ratingOutput = self.rateLinear(hidden)  # [batch_size, 2]
        # ratingOutput = torch.log_softmax(ratingOutput, dim=1)

        return ratingOutput

class cateNet(tnn.Module):

    def __init__(self, embeded_size, hidden, num_layer, dropout, bidirectional):
        """Declare the attributes of class cateNet.
        Args:
            embeded_size: the input_size of torch.nn.LSTM
            hidden: the hidden_size of torch.nn.LSTM
            num_layer: the num_layers of torch.nn.LSTM
            dropout: the dropout rate torch.nn.LSTM
            bidirectional: True or False to declare bidirectional or unidirectional of torch.nn.LSTM
        """
        super(cateNet, self).__init__()
        self.lstm = tnn.LSTM(embeded_size, hidden, num_layer, dropout=dropout, bidirectional=bidirectional)
        self.cateLinear = tnn.Linear(hidden * (2 if bidirectional else 1), 5) # if bidirectional == True: hidden * 2, else: hidden * 1
        # The hidden states of the initial and final time steps are the input to the fully connection layer
        self.w_omega = tnn.Parameter(torch.Tensor(hidden * 2, hidden * 2))
        self.u_omega = tnn.Parameter(torch.Tensor(hidden * 2, 1))

        tnn.init.uniform_(self.w_omega, -0.1, 0.1)
        tnn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, input):
        """Predict a category based on the input via cateNet.
        Args:
            input: shape => [batch_size, sentence_length, dim] => [32, flex, 200].
                Embedded business reviews after packed.
        Returns:
            categoryOutput: shape => [batch_size, 5].
                Predicted category of the reviews (Restaurants, Shopping, Home Services, Health & Medical, Automotive).
        """

        output, (hidden, cell) = self.lstm(input)

        x = tnn.utils.rnn.pad_packed_sequence(output)[0] # x.shape: [batch_size, seq_len, 2 * num_hiddens]
        x = x.permute(1, 0, 2)

        # Attention begin
        u = torch.tanh(torch.matmul(x, self.w_omega)) # u.shape: [batch_size, seq_len, 2 * num_hiddens]
        att = torch.matmul(u, self.u_omega) # att.shape: [batch_size, seq_len, 1]
        att_score = F.softmax(att, dim=1)  # att_score.shape: [batch_size, seq_len, 1]
        scored_x = x * att_score # scored_x.shape: [batch_size, seq_len, 2 * num_hiddens]
        # Attention finish

        feat = torch.sum(scored_x, dim=1) # feat.shape: [batch_size, 2 * num_hiddens]
        
        categoryOutput = self.cateLinear(feat)

        return categoryOutput

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self, embeded_size, hidden, num_layer, dropout, bidirectional):
        """Declare the attributes of this network and import another two neural networks (rateNet, cateNet).
        Args:
            embeded_size: the input_size of torch.nn.LSTM
            hidden: the hidden_size of torch.nn.LSTM
            num_layer: the num_layers of torch.nn.LSTM
            dropout: the dropout rate torch.nn.LSTM
            bidirectional: True or False to declare bidirectional or unidirectional of torch.nn.LSTM
        """
        super(network, self).__init__()
        self.rate = rateNet(embeded_size, hidden, num_layer, dropout, bidirectional)
        self.cate = cateNet(embeded_size, hidden, num_layer, dropout, bidirectional)
        self.dropout = tnn.Dropout(dropout)

    def forward(self, input, length):
        """Predict a rating and a category based on the input via two self-defined neural network models.
        Args:
            input: shape => [batch_size, sentence_length, dim] => [32, flex, 200].
                Embedded business reviews.
            length: shape => [batch_size].
        Returns:
            ratingOutput: shape => [batch_size, 2].
                Predicted rating of the reviews (positive or negative).
            categoryOutput: shape => [batch_size, 5].
                Predicted category of the reviews (Restaurants, Shopping, Home Services, Health & Medical, Automotive). 
        """
        input = self.dropout(input) # Prevent overfitting.
        input = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first=True) # Pack the input to reduce the loss.

        categoryOutput = self.cate(input)  # [batch_size, 5]. Invoke the cateNet to generate the rate prediction.
        ratingOutput = self.rate(input) # [batch_size, 2]. Invoke the rateNet to generate the rate prediction.


        return ratingOutput, categoryOutput

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        """Declare CrossEntropyLoss. """
        super(loss, self).__init__()
        self.loss_rate = tnn.CrossEntropyLoss() # CrossEntropyLoss for rating.
        self.loss_cate = tnn.CrossEntropyLoss() # CrossEntropyLoss for category.
        self.print = False # DEBUG

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        """Calculate the losses for rating and category, and return the mean of the losses.
        Args:
            ratingOutput: Prediction of rating returned from the network.
            categoryOutput: Prediction of category returned from the network.
            ratingTarget: The lable of rating. 
            categoryTarget: The lable of category.
        Returns:
            Mean of the rating loss and category loss.
        """
        match_rate = self.loss_rate(ratingOutput, ratingTarget) # The loss of rating.
        match_cate = self.loss_cate(categoryOutput, categoryTarget) # The loss of category.

        # DEBUG
        if self.print:
            print("rate: ", match_rate.item())
            print("cate: ", match_cate.item())
        
        return torch.mean(match_rate + match_cate)  # The mean of loss(rating) and loss(category).


net = network(EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, BIDIRECTIONAL) # network(200, 100, 2, 0.5, True)
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 1
batchSize = 32
epochs = 20

# Choose different parameters based on different conditions. 
optimiser = toptim.SGD([
    {'params': net.rate.lstm.parameters()},
    {'params': net.rate.rateLinear.parameters(), 'lr': 0.1},
    {'params': net.cate.parameters(), 'lr': 0.6}
], lr=0.5, momentum=0.01)