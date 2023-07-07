
import numpy as np
import sys
import jax
import jax.numpy as jnp
import equinox as eqx


class SineGen(eqx.Module):
    
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)

    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)

    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """
    sampling_rate:int
    sine_amp:float
    harmonic_num:int
    noise_std:float
    voiced_threshold:int
    dim:int
    flag_for_pulse:bool
    key:jnp.ndarray
    def __init__(self,samp_rate=32000,harmonic_num=0,sine_amp=0.1,noise_std=0.003,voiced_threshold=0,flag_for_pulse=False,*,key):
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.key=key

    def _f02uv(self, f0):
        # generate uv signal
        uv = jnp.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1
        
        # initial phase noise (no noise for fundamental component)
        #ini_key = jax.random.split(self.key,2)
        rand_ini = jax.random.uniform(self.key,
           [f0_values.shape[0], f0_values.shape[2]]
        )
        rand_ini=rand_ini.at[:, 0].set(0)
        rad_values=rad_values.at[:, 0, :].set(rad_values[:, 0, :] + rand_ini)

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # for normal case

            # To prevent torch.cumsum numerical overflow,
            # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
            # Buffer tmp_over_one_idx indicates the time step to add -1.
            # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
            tmp_over_one = jnp.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = jnp.zeros_like(rad_values)
            cumsum_shift=cumsum_shift.at[:, 1:, :].set(tmp_over_one_idx * -1.0)

            sines = jnp.sin(
                jnp.cumsum(rad_values + cumsum_shift, axis=1) * 2 * np.pi
            )
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = jnp.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = jnp.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = jnp.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = jnp.cos(i_phase * 2 * np.pi)
        return sines

    def __call__(self, f0):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        f0_buf = jnp.zeros([f0.shape[0], f0.shape[1], self.dim])
        # fundamental component
        f0_buf=f0_buf.at[:, :, 0].set(f0[:, :, 0])
        for idx in np.arange(self.harmonic_num):
            # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
            f0_buf=f0_buf.at[:, :, idx + 1].set(f0_buf[:, :, 0] * (idx + 2))

        # generate sine waveforms
        sine_waves = self._f02sine(f0_buf) * self.sine_amp

        # generate uv signal
        # uv = torch.ones(f0.shape)
        # uv = uv * (f0 > self.voiced_threshold)
        uv = self._f02uv(f0)

        # noise: for unvoiced should be similar to sine_amp
        #        std = self.sine_amp/3 -> max value ~ self.sine_amp
        # .       for voiced regions is self.noise_std
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        rng=jax.random.PRNGKey(1234)
        noise = noise_amp * jax.random.normal(rng,sine_waves.shape)

        # first: set the unvoiced part to 0 by uv
        # then: additive noise
        # jax.debug.print("{}",sine_waves.shape)
        # jax.debug.print("{}",uv.shape)
        # jax.debug.print("{}",noise.shape)
        sine_waves = sine_waves * uv + noise
        return sine_waves



class SourceModuleHnNSF(eqx.Module):
    l_sin_gen:SineGen
    def __init__(self,sampling_rate=32000,sine_amp=0.1,add_noise_std=0.003,voiced_threshod=0,*,key):
        #super(SourceModuleHnNSF, self).__init__()
        harmonic_num = 8
        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod,key=key
        )

        # to merge source harmonics into a single excitation
        #self.l_tanh = nn.tanh()
        self.merge_w=np.asarray([
            -0.1044, -0.4892, -0.4733, 0.4337, -0.2321,
           -0.1889, 0.1315, -0.1002, 0.0590,])
        self.merge_b=np.asarray([-0.2908])

    def __call__(self, x):
        """
        Sine_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        """
        # source for harmonic branch
        #jax.debug.print("{}",x.shape)
        sine_wavs = jax.lax.stop_gradient(self.l_sin_gen(x))
        sine_wavs = jnp.matmul(sine_wavs,jnp.transpose(self.merge_w)) + self.merge_b
        sine_wavs = jnp.expand_dims(sine_wavs,-1)
        sine_merge = jax.nn.tanh(sine_wavs)
        return sine_merge
