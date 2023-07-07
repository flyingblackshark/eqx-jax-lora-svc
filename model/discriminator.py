import numpy as np
import jax
import jax.numpy as jnp
#import flax.linen as nn
import equinox as eqx
from omegaconf import OmegaConf
from .msd import ScaleDiscriminator
from .mpd import MultiPeriodDiscriminator
from .mrd import MultiResolutionDiscriminator


class Discriminator(eqx.Module):
    MRD:MultiResolutionDiscriminator
    MPD:MultiPeriodDiscriminator
    MSD:ScaleDiscriminator
    def __init__(self,hp,key):
        mrdkey,mpdkey,msdkey = jax.random.split(key,3)
        self.MRD = MultiResolutionDiscriminator(hp,mrdkey)
        self.MPD = MultiPeriodDiscriminator(hp,mpdkey)
        self.MSD = ScaleDiscriminator(msdkey)

    def __call__(self, x,train=True):
        r = self.MRD(x,train=train)
        p = self.MPD(x,train=train)
        s = self.MSD(x,train=train)
        return r + p + s


# if __name__ == '__main__':
#     hp = OmegaConf.load('../config/maxgan.yaml')
#     model = Discriminator(hp)

#     x = torch.randn(3, 1, 16384)
#     print(x.shape)

#     output = model(x)
#     for features, score in output:
#         for feat in features:
#             print(feat.shape)
#         print(score.shape)

#     pytorch_total_params = sum(p.numel()
#                                for p in model.parameters() if p.requires_grad)
#     print(pytorch_total_params)
