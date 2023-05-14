"""
Big Data Analytics
Team BANC
Team Members:   Balaji Jayasankar, Aniket Malsane, Nishit Jain, and Carwyn Collinsworth
Associated IDs: 114360535,         115224188,      112680897,       112605735

This code is sampled from Assignment 3 - and by extension,
https://pytorch.org/tutorials/beginner/transformer_tutorial.html 
"""

import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class TransformerModel(nn.Module):
    
    def __init__(self, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        """This method constructs a torch.nn.Module type superclass which is a model for implementing a transformer model for a language modeling task. This is called when a new object of type TransformerModel is instantiated. 

        Args:
            d_model: the expected number of inputs to the TransformerEncoderLayer and PositionalEncoding
            nhead: the number of heads in the multiheadattention model
            d_hid: the hidden dimension of the feedforward network in the TransformerEncoderLayer
            nlayers: number of sub-encoder-layers in the encoder (TransformerEncoder)
            dropout: probability of dropout (setting output to zero) in both the positional encoding and the transformer encoder layer.

        Returns:
            a TransformerModel object, which is an extension of the base nn.Module class pytorch class - an object capable of building NNs from which keeps track of trainable parameters, etc.
        """
        # Instantiate the superclass (nn.Module) such that our model can build upon it.
        super().__init__()
        # Define a variable that specifies the type of module. This is never accessed and is not necessary.
        self.model_type = 'Transformer'
        # Define a neural network component to contain a single self attention and respective feed forward layer (one TransformerEncoderLayer).
        # Added norm_first=True to add batch normalization
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, norm_first=True)
        # Define a consectutive sequence of encoder_layers layers of transformer encoder layers, using our previous definition.
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # Define a decoder as a linear layer to take the output of the attention component and convert it to a specified number of tokens.
        self.decoder = nn.Linear(d_model, 11)
        # Define the expected input size as an attribute of the model for later references.
        self.d_model = d_model

    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output
    
class TransformerModel(nn.Module):
    
    def __init__(self, d_model: int, classes: int, dropout: float = 0.5):
        # Instantiate the superclass (nn.Module) such that our model can build upon it.
        super().__init__()

        self.linear = nn.Linear(d_model, classes)        

    def forward(self, src: Tensor) -> Tensor:
        output = self.linear(src)
        output = torch.sigmoid(output)
        return output