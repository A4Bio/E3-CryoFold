'''
From https://github.com/wolny/pytorch-3dunet.
'''
import torch.nn as nn
from .buildingblocks import ExtResNetBlock, create_encoders, create_decoders
import torch
import pdb


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]

class ResUNet3D4EM(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        basic_module: basic model for the encoder/decoder (DoubleConv, ExtResNetBlock, ....)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
        is_segmentation (bool): if True (semantic segmentation problem) Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped at the end
        testing (bool): if True (testing mode) the `final_activation` (if present, i.e. `is_segmentation=true`)
            will be applied as the last operation during the forward pass; if False the model is in training mode
            and the `final_activation` (even if present) won't be applied; default: False
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, basic_module=ExtResNetBlock, in_channels=1, BB_out_channels=4, CA_out_channels=4, AA_out_channels=21, f_maps=32, layer_order='gcr',
                 num_groups=8, num_levels=5, conv_kernel_size=3, pool_kernel_size=2, conv_padding=1, **kwargs):
        super(ResUNet3D4EM, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"

        # create encoder path
        self.BB_encoders = create_encoders(in_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,num_groups, pool_kernel_size)
        self.CA_encoders = create_encoders(in_channels+BB_out_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,num_groups, pool_kernel_size)
        self.AA_encoders = create_encoders(in_channels+BB_out_channels, f_maps, basic_module, conv_kernel_size, conv_padding, layer_order,num_groups, pool_kernel_size)

        # create decoder path
        self.BB_decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,upsample=True)
        
        self.CA_decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,upsample=True)

        self.AA_decoders = create_decoders(f_maps, basic_module, conv_kernel_size, conv_padding, layer_order, num_groups,upsample=True)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.BB_final_conv = nn.Conv3d(f_maps[0], BB_out_channels, 1)
        self.CA_final_conv = nn.Conv3d(f_maps[0], CA_out_channels, 1)
        self.AA_final_conv = nn.Conv3d(f_maps[0], AA_out_channels, 1)

        self.final_activation=nn.Softmax(dim=1)


    def forward(self, input):
        x = input

        # encoder part
        encoders_features = []
        for encoder in self.BB_encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        # decoder part
        for decoder, encoder_features in zip(self.BB_decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)
        x = self.BB_final_conv(x)
        BB_output = x
        
        CA_AA_input = torch.cat([input,self.final_activation(BB_output)],axis=1)

        x = CA_AA_input
        # encoder part
        encoders_features = []
        for encoder in self.CA_encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]
        
        for decoder, encoder_features in zip(self.CA_decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)
        x = self.CA_final_conv(x)
        CA_output = x

        x = CA_AA_input
        # encoder part
        encoders_features = []
        for encoder in self.AA_encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
        encoders_features = encoders_features[1:]
        for decoder, encoder_features in zip(self.AA_decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)
        x = self.AA_final_conv(x)
        AA_output = x

        return BB_output, CA_output, AA_output


# if __name__ == '__main__':
#     model = ResUNet3D4EM().to('cuda')
#     a=torch.zeros([8,1,64,64,64]).to('cuda')
#     b=model(a)
#     pdb.set_trace()