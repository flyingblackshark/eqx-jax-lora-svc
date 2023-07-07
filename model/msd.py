import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
#import flax.linen as nn
from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import constant as constant_init
from .snake import snake
#from .weightnorm import WeightStandardizedConv
class ScaleDiscriminator(eqx.Module):
    convs:list
    conv_post:eqx.nn.Conv1d
    def __init__(self,key):
        postkey, *bkeys = jax.random.split(key, 7)
        self.convs = [
            eqx.nn.Conv1d(1,16, kernel_size=[15], stride=1,key=bkeys[0]),
            eqx.nn.Conv1d(16,64, kernel_size=[41], stride=4, groups =4,key=bkeys[1]),
            eqx.nn.Conv1d(64,256,kernel_size= [41], stride=4, groups =16,key=bkeys[2]),
            eqx.nn.Conv1d(256,1024, kernel_size=[41], stride=4, groups =64,key=bkeys[3]),
            eqx.nn.Conv1d(1024,1024, kernel_size=[41], stride=4, groups =256,key=bkeys[4]),
            eqx.nn.Conv1d(1024,1024, kernel_size=[5], stride=1,key=bkeys[5]),
        ]
       
        self.conv_post = eqx.nn.Conv1d(1024,1, kernel_size=[3], stride=1,key=postkey)

    def __call__(self, x,train=True):
        fmap = []
        for l in self.convs:
            x = l(x.transpose(0,2,1)).transpose(0,2,1)
            x = jax.nn.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x.transpose(0,2,1)).transpose(0,2,1)
        fmap.append(x)
        x = jnp.reshape(x,[x.shape[0],-1])
        return [(fmap, x)]
