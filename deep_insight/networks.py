"""
Contains neural decoder network objects
"""

# --------------------------------------------------------------

import torch
import torch.backends.cudnn
from torch import nn

# --------------------------------------------------------------


class Standard_Decoder(nn.Module):
    """
    Standard Neural decoder architecture - takes wavelet image sets, returns outputs
    as defined in tg options dict
    """

    def __init__(self, tg, show_summary=True):
        super(Standard_Decoder, self).__init__()
        self.input_shape = tg.input_shape
        self.target_names = list(tg.loss_functions.keys())
        self.conv_order = []

        # first layer is a gaussian noise augmentation to prevent overfitting
        self.gaussian_noise = GaussianNoise()

        # loop over defined number of downsampling layers
        input_channels = self.input_shape[3]
        for nct in range(0, tg.num_convs_tsr):
            setattr(
                self,
                f"conv_tsr_{nct}",
                TimeDistributed(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=tg.filter_size,
                        kernel_size=(tg.kernel_size, tg.kernel_size),
                        stride=(2, 1),
                        padding=(1, 1),
                    )
                ),
            )

            setattr(self, f"conv_tsr_{nct}_activation", getattr(nn, tg.act_conv)())
            input_channels = tg.filter_size
            setattr(
                self,
                f"conv_fr_{nct}",
                TimeDistributed(
                    nn.Conv2d(
                        in_channels=tg.filter_size,
                        out_channels=tg.filter_size,
                        kernel_size=(tg.kernel_size, tg.kernel_size),
                        stride=(1, 2),
                        padding=(1, 1),
                    )
                ),
            )
            setattr(self, f"conv_fr_{nct}_activation", getattr(nn, tg.act_conv)())
            self.conv_order += [
                f"conv_tsr_{nct}",
                f"conv_tsr_{nct}_activation",
                f"conv_fr_{nct}",
                f"conv_fr_{nct}_activation",
            ]

        self.permute = Lambda(lambda x: x.permute((0, 4, 2, 3, 1)))
        self.conv_order.append("permute")

        num_channels = tg.filter_size
        H = self.input_shape[0]
        layer_counter = 0
        while H > 2:
            setattr(
                self,
                f"conv_after_tsr_{layer_counter}",
                TimeDistributed(
                    nn.Conv2d(
                        in_channels=num_channels,
                        out_channels=tg.filter_size * 2,
                        kernel_size=(2, 1),
                        stride=(2, 2),  # ..todo: this is different - why?
                        padding=(1, 0),
                    )
                ),
            )
            setattr(
                self,
                f"conv_after_tsr_{layer_counter}_activation",
                getattr(nn, tg.act_conv)(),
            )
            num_channels = tg.filter_size * 2
            H = round(H / 2)
            self.conv_order += [
                f"conv_after_tsr_{layer_counter}",
                f"conv_after_tsr_{layer_counter}_activation",
            ]
            layer_counter += 1

        # Flatten and fc
        self.flatten = TimeDistributedFlatten()
        self.fc_orders = []
        for key, output in zip(tg.loss_functions.keys(), tg.outputs[0]):
            fc_order = []
            initial_in_channels = 256 * H  # ..todo: should not be hardcoded!
            for d in range(0, tg.num_dense):
                setattr(
                    self,
                    f"target_{key}_fc_{d}",
                    nn.Linear(initial_in_channels, tg.num_units_dense),
                )
                setattr(
                    self, f"target_{key}_fc_{d}_activation", getattr(nn, tg.act_fc)()
                )
                setattr(
                    self, f"target_{key}_fc_{d}_dropout", nn.Dropout(tg.dropout_ratio)
                )
                fc_order += [
                    f"target_{key}_fc_{d}",
                    f"target_{key}_fc_{d}_activation",
                    f"target_{key}_fc_{d}_dropout",
                ]
                initial_in_channels = tg.num_units_dense
            setattr(
                self,
                f"target_{key}_fc_{tg.num_dense}",
                nn.Linear(initial_in_channels, output.shape[1]),
            )
            fc_order.append(f"target_{key}_fc_{tg.num_dense}")
            self.fc_orders.append(fc_order)

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 1, 4, 2, 3)
        x = self.gaussian_noise(x)
        for step_name in self.conv_order:
            x = getattr(self, step_name)(x.float())
        flat_x = self.flatten(x)
        outputs = []
        for fc_order in self.fc_orders:
            x = getattr(self, fc_order[0])(flat_x)
            for step_name in fc_order[1::]:
                x = getattr(self, step_name)(x)
            outputs.append(torch.squeeze(x, 1))
        return outputs

    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.zeros_(layer.bias)
        if hasattr(layer, "weight"):
            nn.init.kaiming_normal_(layer.weight)


# ---------------------------------------------------------------------


class GaussianNoise(nn.Module):
    """
    Gaussian noise regularizer layer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.register_buffer("noise", torch.tensor(0))

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = (
                self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            )
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x


# ---------------------------------------------------------------------


class TimeDistributed(nn.Module):
    """
    Mimics Keras Timedistributed module. Applies same convolutions across 1st dim.
    """

    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        # x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        x_reshape = torch.reshape(
            x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        )

        y = self.module(x_reshape)
        # reshape Y
        y = torch.reshape(
            y, (x.shape[0], x.shape[1], y.shape[1], y.shape[2], y.shape[3])
        )

        return y


# ---------------------------------------------------------------------


class Lambda(nn.Module):
    """
    Mimics Keras lambda module. Applies a function lambd to each layer.
    """

    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


# ---------------------------------------------------------------------


class TimeDistributedFlatten(nn.Module):
    """
    Flattens input's 1st-3rd dimension.
    """

    def __init__(self):
        super(TimeDistributedFlatten, self).__init__()

    def forward(self, x):
        x_reshape = torch.reshape(
            x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        )
        y = torch.flatten(x_reshape, start_dim=1, end_dim=3)
        y = torch.reshape(y, (x.shape[0], x.shape[1], y.shape[1]))
        return y
