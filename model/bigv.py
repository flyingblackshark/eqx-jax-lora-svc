import numpy as np
import jax
import jax.numpy as jnp
#import flax.linen as nn
#from .alias.act import SnakeAlias
import equinox as eqx
from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import constant as constant_init
from .snake import snake
#from .weightnorm import WeightStandardizedConv

class AMPBlock(eqx.Module):
    convs1:list
    convs2:list
    def __init__(self,channels,kernel_size=3,dilation=(1, 3, 5),*,key):
        bkeys = jax.random.split(key,6)
        self.convs1 =[
            eqx.nn.Conv1d(channels,channels, [kernel_size], 1, dilation=dilation[0],key=bkeys[0]),
            eqx.nn.Conv1d(channels,channels, [kernel_size], 1, dilation=dilation[1],key=bkeys[1]),
            eqx.nn.Conv1d(channels,channels, [kernel_size], 1, dilation=dilation[2],key=bkeys[2])
        ]

        self.convs2 = [
            eqx.nn.Conv1d(channels,channels, [kernel_size], 1, dilation=1,key=bkeys[3]),
            eqx.nn.Conv1d(channels,channels, [kernel_size], 1, dilation=1,key=bkeys[4]),
            eqx.nn.Conv1d(channels,channels, [kernel_size], 1, dilation=1,key=bkeys[5])
        ]

    def __call__(self, x,train=True):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = snake(x)
            xt = c1(xt.transpose(0,2,1)).transpose(0,2,1)
            xt = snake(xt)
            xt = c2(xt.transpose(0,2,1)).transpose(0,2,1)
            x = xt + x
        return x
