


import numpy as np
import jax
import jax.numpy as jnp
#import flax.linen as nn
import equinox as eqx
from .nsf import SourceModuleHnNSF
from .bigv import AMPBlock
from jax.nn.initializers import normal as normal_init
from jax.nn.initializers import constant as constant_init
from .snake import snake
#from .weightnorm import WeightStandardizedConvTranspose
# from .nsf import SourceModuleHnNSF
# from .bigv import  AMPBlock#, SnakeAlias

class SpeakerAdapter(eqx.Module):
    epsilon:float
    W_scale:eqx.nn.Linear
    W_bias:eqx.nn.Linear
    def __init__(self,speaker_dim,adapter_dim,epsilon=1e-5,*,key):
        self.epsilon = epsilon
        scale_key,bias_key = jax.random.split(key,2)
        self.W_scale = eqx.nn.Linear(speaker_dim, adapter_dim,key=scale_key)
        self.W_bias = eqx.nn.Linear(speaker_dim,adapter_dim,key=bias_key)


    def __call__(self, x, speaker_embedding):
        x = x.transpose(0,2,1)
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        std = jnp.sqrt(var + self.epsilon)
        y = (x - mean) / std
        scale = self.W_scale(speaker_embedding)
        bias = self.W_bias(speaker_embedding)
        y *= jnp.expand_dims(scale,1)
        y += jnp.expand_dims(bias,1)
        y = y.transpose(0,2,1)
        return y



class Generator(eqx.Module):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    num_kernels:int
    num_upsamples:int
    conv_pre:eqx.nn.Conv
    scale_factor:int
    adapter:list
    noise_convs:list
    ups:list
    resblocks:list
    m_source:SourceModuleHnNSF
    conv_post:eqx.nn.Conv1d
    def __init__(self,hp,key):
        self.num_kernels = len(hp.gen.resblock_kernel_sizes)
        self.num_upsamples = len(hp.gen.upsample_rates)
        # speaker adaper, 256 should change by what speaker encoder you use
        # pre conv
        pre_key,source_key,key = jax.random.split(key,3)
        self.conv_pre = eqx.nn.Conv1d(hp.gen.ppg_channels,hp.gen.upsample_initial_channel,7, 1,key=pre_key)
        # nsf
        self.adapter=[]
        self.noise_convs=[]
        self.ups=[]
        self.resblocks=[]
        self.scale_factor=np.prod(hp.gen.upsample_rates)
        self.m_source = SourceModuleHnNSF(sampling_rate=hp.audio.sampling_rate,key=source_key)
        # transposed conv-based upsamplers. does not apply anti-aliasing
        
        for i, (u, k) in enumerate(zip(hp.gen.upsample_rates, hp.gen.upsample_kernel_sizes)):
            # spk
            speaker_key,key = jax.random.split(key,2)
            self.adapter.append(SpeakerAdapter(
                256, hp.gen.upsample_initial_channel // (2 ** (i + 1)),key=speaker_key))
            # print(f'ups: {i} {k}, {u}, {(k - u) // 2}')
            # base
            ups_key,key = jax.random.split(key,2)
            self.ups.append(
                    eqx.nn.ConvTranspose1d(
                        hp.gen.upsample_initial_channel // (2 ** i),
                        hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        k,u,key=ups_key)
            )
            # nsf
            noise_key,key = jax.random.split(key,2)
            if i + 1 < len(hp.gen.upsample_rates):
                stride_f0 = np.prod(hp.gen.upsample_rates[i + 1:])
                stride_f0 = int(stride_f0)
                self.noise_convs.append(
                    eqx.nn.Conv1d(
                        1,
                        hp.gen.upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        key=noise_key
                    )
                )
            else:
                self.noise_convs.append(
                    eqx.nn.Conv1d(1,hp.gen.upsample_initial_channel //
                           (2 ** (i + 1)), kernel_size=1,key=noise_key)
                )

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        for i in range(len(self.ups)):
            ch = hp.gen.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(hp.gen.resblock_kernel_sizes, hp.gen.resblock_dilation_sizes):
                amp_key,key = jax.random.split(key,2)
                self.resblocks.append(AMPBlock(ch, k, d,key=amp_key))

        # post conv
        #self.activation_post = nn.leaky_relu(ch)
        post_key,key = jax.random.split(key,2)
        self.conv_post = eqx.nn.Conv1d(ch , 1, 7, 1, use_bias=False,key=post_key)
        # weight initialization
    def __call__(self, spk, x, f0, train=True):
        #rng = jax.random.PRNGKey(1234)
        # nsf
        f0 = f0[:, None]
        B, H, W = f0.shape
        f0 = jax.image.resize(f0, shape=(B, H, W * self.scale_factor), method='nearest').transpose(0,2,1)
        #f0 = self.f0_upsamp(f0).transpose(1, 2)
        har_source = self.m_source(f0)
        har_source = har_source.transpose(0,2,1)
        # pre conv
        # if train:
        #     #x = x + torch.randn_like(x)     # Perturbation
        #     x = x + jax.random.normal(rng,x.shape)  
            
        x = x.transpose(0,2,1)      # [B, D, L]
        x = self.conv_pre(x.transpose(0,2,1)).transpose(0,2,1)

        x = x * jax.nn.tanh(jax.nn.softplus(x))

        for i in range(self.num_upsamples):
            # upsampling
            x = self.ups[i](x.transpose(0,2,1)).transpose(0,2,1)

            # adapter
            x = self.adapter[i](x, spk)
            # nsf
            x_source = self.noise_convs[i](har_source.transpose(0,2,1)).transpose(0,2,1)
            x = x + x_source
            
            
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x,train=train)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x,train=train)
            x = xs / self.num_kernels
        # post conv
        #x = self.activation_post(x)
        #x = nn.leaky_relu(x)
        x = snake(x)
        #x = self.norms2(x,use_running_average=not train)
        x = self.conv_post(x.transpose(0,2,1)).transpose(0,2,1)
        x = jax.nn.tanh(x)
        return x

    # def remove_weight_norm(self):
    #     for l in self.ups:
    #         remove_weight_norm(l)
    #     for l in self.resblocks:
    #         l.remove_weight_norm()
    #     remove_weight_norm(self.conv_pre)

    # def eval(self, inference=False):
    #     super(Generator, self).eval()
    #     # don't remove weight norm while validation in training loop
    #     if inference:
    #         self.remove_weight_norm()

    def inference(self, spk, ppg, f0):
        MAX_WAV_VALUE = 32768.0
        audio = self.forward(spk, ppg, f0, False)
        audio = audio.squeeze()  # collapse all dimension except time axis
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()
        return audio

    def pitch2wav(self, f0):
        MAX_WAV_VALUE = 32768.0
        # nsf
        f0 = f0[:, None]
        f0 = self.f0_upsamp(f0).transpose(1, 2)
        har_source = self.m_source(f0)
        audio = har_source.transpose(1, 2)
        audio = audio.squeeze()  # collapse all dimension except time axis
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()
        return audio

    def train_lora(self):
        print("~~~train_lora~~~")
        for p in self.parameters():
           p.requires_grad = False
        for p in self.adapter.parameters():
           p.requires_grad = True
        for p in self.resblocks.parameters():
           p.requires_grad = True