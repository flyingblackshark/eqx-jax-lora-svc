import numpy as np
import jax
import jax.numpy as jnp
#import flax.linen as nn
import equinox as eqx
from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import constant as constant_init
from .snake import snake
#from .weightnorm import WeightStandardizedConv
import scipy
class DiscriminatorR(eqx.Module):
    convs: list
    resolution:list
    hp:list
    LRELU_SLOPE:float
    conv_post:eqx.nn.Conv2d
    def __init__(self,resolution,hp,key):
        self.resolution = resolution
        self.hp = hp 
        self.LRELU_SLOPE = self.hp.mpd.lReLU_slope
        postkey, *bkeys = jax.random.split(key, 6)
        self.convs = [
            eqx.nn.Conv2d(1,32, kernel_size=(3, 9),key=bkeys[0]),
            eqx.nn.Conv2d( 32,32, kernel_size=(3, 9), stride=(1, 2),key=bkeys[1]),
            eqx.nn.Conv2d( 32,32, kernel_size=(3, 9), stride=(1, 2),key=bkeys[2]),
            eqx.nn.Conv2d( 32,32, kernel_size=(3, 9), stride=(1, 2),key=bkeys[3]),
            eqx.nn.Conv2d( 32,32, kernel_size=(3, 3),key=bkeys[4]),
        ]
        self.conv_post = eqx.nn.Conv2d( 32,1, kernel_size=(3, 3),key=postkey)
        
        

    def __call__(self, x,train=True):
        fmap = []

        x = self.spectrogram(x)
        x = jnp.expand_dims(x,1)
        for l in self.convs:
            x = l(x.transpose(0,2,3,1)).transpose(0,3,1,2)
            x = jax.nn.leaky_relu(x, self.LRELU_SLOPE)
            #x = snake(x)
            fmap.append(x)
        x = self.conv_post(x.transpose(0,2,3,1)).transpose(0,3,1,2)
        fmap.append(x)
        x = jnp.reshape(x, [x.shape[0],-1])

        return fmap, x

    def spectrogram(self, x):
       
        n_fft, hop_length, win_length = self.resolution
        hann_win = scipy.signal.get_window('hann', n_fft)
        scale = np.sqrt(1.0 / hann_win.sum()**2)
        x = jnp.pad(x, [(0,0),(0,0),(int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2))], mode='reflect')
        x = x.squeeze(1)
        x = jax.scipy.signal.stft(x,fs=32000, nfft=n_fft, noverlap=win_length-hop_length, nperseg=win_length) #[B, F, TT, 2]
        mag = jnp.abs(x[2]/scale)
       
        return mag


class MultiResolutionDiscriminator(eqx.Module):
    discriminators:list
    def __init__(self,hp,key):
        resolutions = eval(hp.mrd.resolutions)
        bkeys = jax.random.split(key, len(resolutions))
        self.discriminators = [DiscriminatorR(resolution,hp,key=bkey) for (resolution,bkey) in zip(resolutions,bkeys)]
        

    def __call__(self, x,train=True):
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x,train=train))

        return ret  # [(feat, score), (feat, score), (feat, score)]
