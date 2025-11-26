import numpy as np
import torch
from torch import nn
from torch.nn.functional import elu
import pytorch_lightning as pl

import sys 
sys.path.append('G:/ninon_workspace/imagery2024/2D_Reconstruction/Generation_2D/reconstruction/code/')

from dc_ldm.models.base_model import base_model

"""
Note: this code is adapted from https://github.com/braindecode/braindecode/tree/master/braindecode
which is the implemented EEGNet version from torcheeg.

The Convolutional Classifier was removed for a Linear Layer Classifier which facilitates the embedding goal.
"""

def get_activation(activation_name):
    """Returns the activation function based on the name."""
    activations = {
        'elu': nn.functional.elu,
        'relu': nn.functional.relu,
        'leaky_relu': lambda x: nn.functional.leaky_relu(x, negative_slope=0.1),
        'mish': lambda x: x * torch.tanh(nn.functional.softplus(x))
    }
    return activations.get(activation_name, nn.functional.elu)


class EEGNet_Embedding(base_model):
    """EEGNet v4 model from Lawhern et al 2018.

    Parameters
    ----------
    in_chans : int

    Notes
    -----
    This implementation is not guaranteed to be correct, has not been checked
    by original authors, only reimplemented from the paper description.

    References
    ----------
    .. [EEGNet4] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon,
       S. M., Hung, C. P., & Lance, B. J. (2018).
       EEGNet: A Compact Convolutional Network for EEG-based
       Brain-Computer Interfaces.
       arXiv preprint arXiv:1611.08024.
    """

    def __init__(
        self,
        #General
        lr = 1e-3,
        one_cycle_lr = True,
        weight_decay = 0.0,
        epochs = 100,
        in_chans = 61,
        n_classes = 6,
        final_conv_length="auto",
        input_window_samples=None,
        #Convolutions and Pooling (depth of spatial conv = F2*D)
        F1=8,
        D=2,
        pool_mode="mean",
        kernel_length=64,
        depthwise_kernel_length=32,
        separable_kernel_length=32,
        activation='elu',
        drop_prob=0.25,
        momentum = 0.01,
        max_lr = None,
        three_phase = False,
        pct_start = 0.3,
        beta1=0.9,
        beta2=0.999, 
        eps=1e-8,
        **kwargs
    ):
        super().__init__()
        if final_conv_length == "auto":
            assert input_window_samples is not None

        self.max_lr = max_lr if max_lr is not None else None #self.lr * 10
        self.pct_start = pct_start
        self.three_phase = three_phase
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps

        self.in_chans = in_chans
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.final_conv_length = final_conv_length
        self.pool_mode = pool_mode
        self.F1 = F1
        self.D = D
        self.F2 = D*F1
        self.kernel_length = kernel_length
        self.depthwise_kernel_length = depthwise_kernel_length
        self.separable_kernel_length = separable_kernel_length
        self.drop_prob = drop_prob
        self.momentum = momentum
        self.one_cycle_lr = one_cycle_lr
        self.lr = lr
        self.weight_decay = weight_decay
        self.activation_name = activation
        self.activation_fn = get_activation(activation)
        pool_class = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[self.pool_mode]
        self.loss = nn.NLLLoss() #EEGNet uses a log softmax layer. Therefore, using a NLLLoss() equates to CrossEntropyLoss()
        self.epochs = epochs

        self.ensuredims = Ensure4d() # from [b c t] to [b c t 1]
        self.dimshuffle = Expression(_transpose_to_b_1_c_0) # from [b c t 1] to [b 1 c t] --> first conv over temporal dim with kernel_length
        self.conv_temporal = nn.Conv2d(
            in_channels = 1,
            out_channels = self.F1,
            kernel_size = (1, self.kernel_length),
            stride = 1,
            bias = False,
            padding = (0, self.kernel_length // 2),
            )
        self.bnorm_temporal = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
        self.conv_spatial = Conv2dWithConstraint(
            self.F1,
            self.F1 * self.D,
            (self.in_chans, 1),
            max_norm=1,
            stride=1,
            bias=False,
            groups=self.F1,
            padding=(0, 0),
            )
        self.bnorm_1 = nn.BatchNorm2d(self.F1 * self.D, momentum=self.momentum, affine=True, eps=1e-3)
        self.elu_1 = Expression(self.activation_fn)
        self.pool_1 = pool_class(kernel_size=(1, 4), stride=(1, 4))
        #print('DROPOUT = ', self.drop_prob)
        self.drop_1 = nn.Dropout(p=self.drop_prob)
        self.conv_separable_depth = nn.Conv2d(
            self.F1 * self.D,
            self.F1 * self.D,
            (1, self.depthwise_kernel_length),
            stride=1,
            bias=False,
            groups=self.F1 * self.D,
            padding=(0, self.depthwise_kernel_length // 2),
            )
        self.conv_separable_point = nn.Conv2d(
            self.F1 * self.D,
            self.F2,
            (1, self.separable_kernel_length),
            stride=1,
            bias=False,
            padding=(0, self.separable_kernel_length // 2),
            )
        self.bnorm_2 = nn.BatchNorm2d(self.F2, momentum=self.momentum, affine=True, eps=1e-3)
        self.elu_2 = Expression(self.activation_fn)
        self.pool_2 = pool_class(kernel_size=(1, 8), stride=(1, 8))
        self.drop_2 = nn.Dropout(p=self.drop_prob)

        #The following tests the output dimensions to pass the right dimensions to the classifier head
        out = self.partial_forward(
                torch.ones(
                (1, self.in_chans, self.input_window_samples, 1),
                dtype=torch.float32
                )
            )
        n_out_virtual_chans = out.cpu().data.numpy().shape[2]
        if self.final_conv_length == "auto":
            n_out_time = out.cpu().data.numpy().shape[3]
            self.final_conv_length = n_out_time

        # What is different (Embedding layer) :
        dummy_input = torch.ones((1, self.in_chans, self.input_window_samples, 1), dtype=torch.float32)
        dummy_output = self.partial_forward(dummy_input)
        output_size = dummy_output.flatten(start_dim=1).shape[1] # calculate output size for the FC layer
        self.embedding = nn.Linear(output_size, 512) # Dynamic output size
        print("SHAPE EMBEDDING ", output_size, self.F2*15)
        #self.embedding = None #nn.Linear(self.F2*15, 512)

        #print("dummy_input shape:", dummy_input.shape)
        #print("dummy_output shape:", dummy_output.shape)

        self.classifier = nn.Linear(512, self.n_classes)

        self.softmax = nn.LogSoftmax(dim=1)
        # Transpose time back in third dimension (axis=2)
        self.permute_back = Expression(_transpose_1_0)
        self.squeeze = Expression(squeeze_final_output)
        _glorot_weight_zero_bias(self) #Initialize weights

    def partial_forward(self, x): #for a sample of [1 8 500] and kernel_size=128
        #Used to initially determine the input dimensions to the classifier head
        x = self.ensuredims(x) # [1 8 500 1]
        x = self.dimshuffle(x) # [1 1 8 500]
        x = self.conv_temporal(x) # [1 8 8 501]
        x = self.bnorm_temporal(x) # [1 8 8 501]
        x = self.elu_1(x) # [1 8 8 501]
        x = self.conv_spatial(x) # [1 16 1 501]
        x = self.bnorm_1(x) # [1 16 1 501]
        x = self.elu_1(x) # [1 16 1 501]
        x = self.pool_1(x) # [1 16 1 125]
        x = self.drop_1(x) # [1 16 1 125]
        x = self.conv_separable_depth(x) # [1 16 1 126]
        x = self.conv_separable_point(x) # [1 16 1 126]
        x = self.bnorm_2(x) # [1 16 1 126]
        x = self.elu_2(x) # [1 16 1 126]
        x = self.pool_2(x) # [1 16 1 15]
        x = self.drop_2(x) # Sven [1 16 1 15] vs us after param tuning [1, 64, 1, 128]
        #print("end of partial foward shape", x.shape)
        return x

    def forward(self, x, return_embedding = False):


        x = self.partial_forward(x) # [1 16 1 15] #bs x F2 x 1 x 15 (128*15)
        #print("Output shape from partial_forward:", x.shape)
        x = x.flatten(start_dim=1) # bs x F2*15
        #print("Shape before embedding:", x.shape)
        # Define embedding layer dynamically on first run to get proper shape
        if self.embedding is None:
            print("embedding None")
            self.embedding = nn.Linear(x.shape[1], 512).to(x.device)

        #print(f"Before embedding layer: {x.shape}")

        embedding  = self.embedding(x) # bs x 1024

        #print(f"After embedding layer: {x.shape}")

        if return_embedding:
            return embedding  # Return the 512-dim embedding
        else:
            # For classification, continue with classifier
            classification = self.classifier(embedding)  # [B, 6]
            classification = self.softmax(classification)
            return classification
    
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay = self.weight_decay)
    #     return optimizer
    def configure_optimizers(self):

        print("OPtim embedding")
        #optimizer = torch.optim.Adam(params = self.parameters(), lr = self.lr, weight_decay = self.weight_decay)

        optimizer = torch.optim.Adam(
            params=self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay,
            betas=(self.beta1, self.beta2),
            eps=self.eps
        )

        div_factor = self.max_lr / 1e-6 

        if self.one_cycle_lr:
            one_cycle = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer = optimizer,
                    #max_lr = self.lr,
                    max_lr = self.max_lr,  
                    pct_start = self.pct_start,
                    three_phase = self.three_phase,
                    steps_per_epoch = len(self.trainer.datamodule.train_dataloader()), #* self.epochs,
                    epochs = self.epochs,
                    div_factor= div_factor
                    # steps_per_epoch = self.trainer.estimated_stepping_batches // self.epochs,
                    #cycle_momentum = True
                    )
            
            lr_scheduler = {
                "scheduler": one_cycle, #lr
                "interval": "step",
                "name": "Learning Rate Scheduling"
            }
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]
      
#Helper functions and classes
def _transpose_to_b_1_c_0(x):
    return x.permute(0, 3, 1, 2)

def _transpose_1_0(x):
    return x.permute(0, 1, 3, 2)

def _glorot_weight_zero_bias(model):
    """Initalize parameters of all modules by initializing weights with
    glorot
     uniform/xavier initialization, and setting biases to zero. Weights from
     batch norm layers are set to 1.

    Parameters
    ----------
    model: Module
    """
    for module in model.modules():
        if hasattr(module, "weight") and not "NLLLoss" in module.__class__.__name__:
            if not ("BatchNorm" in module.__class__.__name__):
                nn.init.xavier_uniform_(module.weight, gain=1)
            else:
                nn.init.constant_(module.weight, 1)
        if hasattr(module, "bias"):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

def squeeze_final_output(x):
    """Removes empty dimension at end and potentially removes empty time
     dimension. It does  not just use squeeze as we never want to remove
     first dimension.

    Returns
    -------
    x: torch.Tensor
        squeezed tensor
    """
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x

class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d with weight constraint."""
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)
    
class Expression(nn.Module):
    """Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn : callable
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if hasattr(self.expression_fn, "func") and hasattr(
            self.expression_fn, "kwargs"
        ):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__, str(self.expression_fn.kwargs)
            )
        elif hasattr(self.expression_fn, "__name__"):
            expression_str = self.expression_fn.__name__
        else:
            expression_str = repr(self.expression_fn)
        return (
            self.__class__.__name__ +
            "(expression=%s) " % expression_str
        )
    
class Ensure4d(nn.Module):
    def forward(self, x):
        while len(x.shape) < 4:
            x = x.unsqueeze(-1)
        return x
