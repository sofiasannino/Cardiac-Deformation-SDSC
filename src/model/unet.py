from typing import Union, Type, List, Tuple

from copy import deepcopy
from typing import Literal, Sequence
from torch import nn
import torch
from dataclasses import dataclass
import torch

from building_blocks.helper import convert_conv_op_to_dim
from building_blocks.plain_conv_encoder import PlainConvEncoder
from building_blocks.residual import BasicBlockD, BottleneckD
from building_blocks.residual_encoders import ResidualEncoder
from building_blocks.unet_decoder import UNetDecoder, CoordHead
from building_blocks.unet_residual_decoder import UNetResDecoder
from initialization.weight_init import InitWeights_He
from initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd



class AbstractDynamicNetworkArchitectures(nn.Module):

    def __init__(self):
        super(AbstractDynamicNetworkArchitectures, self).__init__()
        # Key to the position holding all the encoder weights
        self.key_to_encoder: str
        # Key to the full stem -- Can be located within or outside the encoder
        self.key_to_stem: str
        # Not sure yet if we need anything but this -- but minor redundancy is okay I suppose
        # Key to the weights that are dependent on the input channels.
        #   Can hold multiple weights (e.g. for bad weight mappings like in this repo >.<' )
        self.keys_to_in_proj: Sequence[str] # this stores the path to the first convolution layer
        self.key_to_lpe: str | None = None  # LPE == Learnable Positional Embedding


class PlainConvUNet(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        input_channels: int, # number of input channels
        n_stages: int, # number of resolution stages 
        features_per_stage: Union[int, List[int], Tuple[int, ...]], # feature channels per stage
        conv_op: Type[_ConvNd], # type of convolution 3d/2d
        kernel_sizes: Union[int, List[int], Tuple[int, ...]], # convolution kernels
        strides: Union[int, List[int], Tuple[int, ...]], # convolution stride
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]], # number of convolution per encoder stage
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]], # always n_stages -1 stages in decoder
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
    ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()

        self.key_to_encoder = "encoder.stages"  # Contains the stem as well.
        self.key_to_stem = "encoder.stages.0"
        self.keys_to_in_proj = (
            "encoder.stages.0.0.convs.0.all_modules.0",
            "encoder.stages.0.0.convs.0.conv",  # duplicate of above
        )

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages # creates [2, 2, 2, 2, 2, 2]
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1) # creates [2, 2, 2, 2, 2]
        assert len(n_conv_per_stage) == n_stages, (
            "n_conv_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_conv_per_stage: {n_conv_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )
        # encoder 
        self.encoder = PlainConvEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True, # return intermediate features for skip connections [f0, f1, ..., fstages]
            nonlin_first=nonlin_first,
        )
        self.decoder = UNetDecoder( # starts from the bottleneck and upsample, to remove deepsupervision, change everything for coord regression
            self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision, nonlin_first=nonlin_first
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips) 

    def compute_conv_feature_map_size(self, input_size): # estimation of how many convolutional feature-map elements are created for a given input size
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module) # standard He initialization 

class PlainConvUNetCoord(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        input_channels: int, # number of input channels
        n_stages: int, # number of resolution stages 
        features_per_stage: Union[int, List[int], Tuple[int, ...]], # feature channels per stage
        conv_op: Type[_ConvNd], # type of convolution 3d/2d
        kernel_sizes: Union[int, List[int], Tuple[int, ...]], # convolution kernels
        strides: Union[int, List[int], Tuple[int, ...]], # convolution stride
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]], # number of convolution per encoder stage
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]], # always n_stages -1 stages in decoder
        pool_size : int , # size of avg pooling in coord head, added
        hidden_coord: int, # hidden dimension of linear layer coord head, added
        K : int, # number of wanted control points
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        final_activation: str | None = None,  # None, "sigmoid", or "tanh"
    ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()

        self.key_to_encoder = "encoder.stages"  # Contains the stem as well.
        self.key_to_stem = "encoder.stages.0"
        self.keys_to_in_proj = (
            "encoder.stages.0.0.convs.0.all_modules.0",
            "encoder.stages.0.0.convs.0.conv",  # duplicate of above
        )

        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages # creates [2, 2, 2, 2, 2, 2]
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1) # creates [2, 2, 2, 2, 2]
        assert len(n_conv_per_stage) == n_stages, (
            "n_conv_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_conv_per_stage: {n_conv_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )
        # encoder 
        self.encoder = PlainConvEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_conv_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            return_skips=True, # return intermediate features for skip connections [f0, f1, ..., fstages]
            nonlin_first=nonlin_first,
        )
        self.decoder = CoordHead( # starts from the bottleneck and add a fully connected network for coord regression
            self.encoder, pool_size=pool_size, hidden_coord=hidden_coord, K=K, final_activation=final_activation,
        )

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips) # coords 

    def compute_conv_feature_map_size(self, input_size): # estimation of how many convolutional feature-map elements are created for a given input size
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module) # standard He initialization 


