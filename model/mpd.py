# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.utils import weight_norm, spectral_norm
import numpy as np
import jax
import jax.numpy as jnp
#import flax.linen as nn
import equinox as eqx
from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import constant as constant_init
from .snake import snake
#from .weightnorm import WeightStandardizedConv
class DiscriminatorP(eqx.Module):
    LRELU_SLOPE:float
    period:tuple
    convs:list
    conv_post:eqx.nn.Conv2d
    def __init__(self,hp,period,key):
        self.LRELU_SLOPE = hp.mpd.lReLU_slope
        self.period = period
        kernel_size = hp.mpd.kernel_size
        stride = hp.mpd.stride
      
        postkey, *bkeys = jax.random.split(key, 6)
        self.convs = [
            eqx.nn.Conv2d(1,64, (kernel_size, 1), (stride, 1),key=bkeys[0],padding=(kernel_size // 2, 0)),
            eqx.nn.Conv2d(64,128, (kernel_size, 1), (stride, 1),key=bkeys[1],padding=(kernel_size // 2, 0)),
            eqx.nn.Conv2d(128,256, (kernel_size, 1), (stride, 1),key=bkeys[2],padding=(kernel_size // 2, 0)),
            eqx.nn.Conv2d(256,512, (kernel_size, 1), (stride, 1),key=bkeys[3],padding=(kernel_size // 2, 0)),
            eqx.nn.Conv2d(512,1024, (kernel_size, 1), 1,key=bkeys[4],padding=(kernel_size // 2, 0)),
        ]
        self.conv_post = eqx.nn.Conv2d(1024,1, (3, 1), 1,key=postkey, padding=(1, 0))

    def __call__(self, x,train=True):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = jnp.pad(x, [(0,0),(0,0),(0, n_pad)], "reflect")
            t = t + n_pad
        x = jnp.reshape(x,[b, c, t // self.period, self.period])

        for l in self.convs:
            x = l(x.transpose(0,2,3,1)).transpose(0,3,1,2)
            x = jax.nn.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x.transpose(0,2,3,1)).transpose(0,3,1,2)
        fmap.append(x)
        #x = torch.flatten(x, 1, -1)
        x = jnp.reshape(x,[x.shape[0],-1])

        return fmap, x


class MultiPeriodDiscriminator(eqx.Module):
    discriminators:list
    def __init__(self,hp,key):
       # super(MultiPeriodDiscriminator, self).__init__()
        bkeys = jax.random.split(key, len(hp.mpd.periods))
        self.discriminators = [DiscriminatorP(hp, period,key=bkey) for (period,bkey) in zip(hp.mpd.periods,bkeys)]
        

    def __call__(self, x,train=True):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x,train=train))

        return ret  # [(feat, score), (feat, score), (feat, score), (feat, score), (feat, score)]
