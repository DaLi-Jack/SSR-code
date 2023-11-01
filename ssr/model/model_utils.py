from .encoder import SpatialEncoder, ImageEncoder
from torch import nn
import functools


def make_encoder(conf, **kwargs):
    enc_type = conf['encoder_type']  # spatial | global
    if enc_type == "spatial":
        net = SpatialEncoder.from_conf(conf, **kwargs)
    elif enc_type == "global":
        net = ImageEncoder.from_conf(conf, **kwargs)
    else:
        raise NotImplementedError("Unsupported encoder type")
    return net
