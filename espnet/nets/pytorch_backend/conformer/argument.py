# Copyright 2020 Hirofumi Inaguma
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer common arguments."""


from distutils.util import strtobool


def add_arguments_conformer_common(group):
    """Add Transformer common arguments."""
    group.add_argument(
        "--transformer-encoder-pos-enc-layer-type",
        type=str,
        default="abs_pos",
        choices=["abs_pos", "scaled_abs_pos", "rel_pos"],
        help="transformer encoder positional encoding layer type",
    )
    group.add_argument(
        "--transformer-encoder-activation-type",
        type=str,
        default="swish",
        choices=["relu", "hardtanh", "selu", "swish"],
        help="transformer encoder activation function type",
    )
    group.add_argument(
        "--macaron-style",
        default=False,
        type=strtobool,
        help="Whether to use macaron style for positionwise layer",
    )
    # CNN module
    group.add_argument(
        "--use-cnn-module",
        default=False,
        type=strtobool,
        help="Use convolution module or not",
    )
    group.add_argument(
        "--cnn-module-kernel",
        default=31,
        type=int,
        help="Kernel size of convolution module.",
    )
    return group

def add_arguments_dense_conformer_common(group):
    group = add_arguments_conformer_common(group)

    group.add_argument(
        "--aunits",
        default=2048,
        type=int,
        help="number of units of auxiliary encoder"
    )
    group.add_argument(
        "--alayers",
        default=6,
        type=int,
        help="number of layers of auxiliary encoder",
    )
    group.add_argument(
        "--encoder-fusion",
        default=False,
        type=strtobool,
        help="is add one more attention to fuse two encoders",
    )
    group.add_argument(
        "--encoder-share-weights",
        default=False,
        type=strtobool,
        help="is sharing encoders' weights",
    )
    group.add_argument(
        "--decoder-fusion-type",
        default="vanilla",
        type=str,
        choices=["vanilla", "stacked", "gate"],
        help="type of decoder",
    )
    return group