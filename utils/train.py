import os
import time
import logging
import math
import tqdm
import itertools
import traceback
import flax
import jax
import optax
import numpy as np
import orbax
#from flax import linen as nn
from utils.dataloader import create_dataloader

from utils.writer import MyWriter
from utils.stft import TacotronSTFT
from utils.stft_loss import MultiResolutionSTFTLoss
from model.generator import Generator
from model.discriminator import Discriminator
import jax.numpy as jnp
from functools import partial
from typing import Any, Tuple
import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
import jax.random as jr
import jax.sharding as sharding
import numpy as np
import optax  # https://github.com/deepmind/optax

PRNGKey = jnp.ndarray

num_devices = len(jax.devices())
devices = mesh_utils.create_device_mesh((num_devices, 1))
shard = sharding.PositionalSharding(devices)
def train(rank, args, chkpt_path, hp, hp_str):
    num_devices = jax.device_count()
    total_steps = 200000
    # #@partial(jax.pmap, static_broadcasted_argnums=(1))
    # def create_generator_state(rng, model_cls): 
    #     r"""Create the training state given a model class. """ 
    #     model = model_cls(hp=hp)
    #     exponential_decay_scheduler = optax.exponential_decay(init_value=hp.train.adam.lr, transition_steps=total_steps,decay_rate=hp.train.lr_decay, transition_begin=0,staircase=False)
    #     tx = optax.lion(learning_rate=exponential_decay_scheduler, b1=hp.train.adam.beta1,b2=hp.train.adam.beta2)
    #     fake_ppg = jnp.ones((hp.train.batch_size,40,1280))
    #     fake_pit = jnp.ones((hp.train.batch_size,40))
    #     fake_spk = jnp.ones((hp.train.batch_size,256))

    #     variables = model.init(rng, fake_spk,fake_ppg,fake_pit,train=False)

    #     state = TrainState.create(apply_fn=model.apply, tx=tx, params=variables['params'])
        
    #     return state
    # #@partial(jax.pmap, static_broadcasted_argnums=(1))
    # def create_discriminator_state(rng, model_cls): 
    #     r"""Create the training state given a model class. """ 
    #     model = model_cls(hp=hp)
    #     fake_audio = jnp.ones((hp.train.batch_size,1,12800))
    #     exponential_decay_scheduler = optax.exponential_decay(init_value=hp.train.adam.lr, transition_steps=total_steps,decay_rate=hp.train.lr_decay, transition_begin=0,staircase=False)
    #     tx = optax.lion(learning_rate=exponential_decay_scheduler, b1=hp.train.adam.beta1,b2=hp.train.adam.beta2)
    #     variables = model.init(rng, fake_audio,train=False)
    #     state = TrainState.create(apply_fn=model.apply, tx=tx,params=variables['params'])
        
    #     return state
    def g_loss_fn(params_g,static_g,discriminator_model:Discriminator,ppg : jnp.ndarray  , pit : jnp.ndarray, spk : jnp.ndarray ,audio:jnp.ndarray):
        model_g = eqx.combine(params_g, static_g)
        stft = TacotronSTFT(filter_length=hp.audio.filter_length,
                    hop_length=hp.audio.hop_length,
                    win_length=hp.audio.win_length,
                    n_mel_channels=hp.audio.n_mel_channels,
                    sampling_rate=hp.audio.sampling_rate,
                    mel_fmin=hp.audio.mel_fmin,
                    mel_fmax=hp.audio.mel_fmax)
        stft_criterion = MultiResolutionSTFTLoss(eval(hp.mrd.resolutions))
        fake_audio = model_g(spk,ppg,pit)
        mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
        mel_real = stft.mel_spectrogram(audio.squeeze(1))
        mel_loss = jnp.mean(optax.huber_loss(mel_fake, mel_real))* hp.train.mel_lamb
        #Multi-Resolution STFT Loss
        sc_loss, mag_loss = stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
        stft_loss = (sc_loss + mag_loss) * hp.train.stft_lamb
        # Generator Loss 
        disc_fake = discriminator_model(fake_audio)
        score_loss = 0.0
        for (_, score_fake) in disc_fake:
            score_loss += jnp.mean(jnp.square(score_fake - 1.0))
        score_loss = score_loss / len(disc_fake)
        # Feature Loss
        disc_real= discriminator_model(audio)
        feat_loss = 0.0
        for (feat_fake, _), (feat_real, _) in zip(disc_fake, disc_real):
            for fake, real in zip(feat_fake, feat_real):
                feat_loss += jnp.mean(jnp.abs(fake - real))
        feat_loss = feat_loss / len(disc_fake)
        feat_loss = feat_loss * 2
        loss_g = mel_loss + score_loss +  feat_loss + stft_loss

        return loss_g, (fake_audio,mel_loss,stft_loss)
    def d_loss_fn(params_d,static_d,audio,fake_audio):
        model_d = eqx.combine(params_d, static_d)
        disc_fake  = model_d(fake_audio)
        disc_real = model_d(audio)
        loss_d = 0.0
        for (_, score_fake), (_, score_real) in zip(disc_fake, disc_real):
            loss_d += jnp.mean(jnp.square(score_real - 1.0))
            loss_d += jnp.mean(jnp.square(score_fake))
        loss_d = loss_d / len(disc_fake)
        
        return loss_d
    
    def combine_step(generator_model: Generator, optim_g:optax.GradientTransformation,discriminator_model: Discriminator,optim_d:optax.GradientTransformation,
                       ppg : jnp.ndarray  , pit : jnp.ndarray, spk : jnp.ndarray ,audio:jnp.ndarray):
      
        
        params_g,static_g = eqx.partition(generator_model, eqx.is_array)
        grad_fn_g = jax.value_and_grad(g_loss_fn, has_aux=True)
        (loss_g,(fake_audio,mel_loss,stft_loss)), grads_g = grad_fn_g(params_g,static_g,discriminator_model,ppg,pit,spk,audio)
        updates_g, optim_g = optim_g.update(grads_g, optim_g, generator_model)
        generator_model = eqx.apply_updates(generator_model, updates_g)
        # Average across the devices.
        # grads_g = jax.lax.pmean(grads_g, axis_name='num_devices')
        # loss_g = jax.lax.pmean(loss_g, axis_name='num_devices')
        loss_m = mel_loss
        loss_s = stft_loss
        # Generate data with the Generator, critique it with the Discriminator.
        grad_fn_d = jax.value_and_grad(d_loss_fn, has_aux=False)
        params_d,static_d = eqx.partition(discriminator_model, eqx.is_array)
        loss_d, grads_d = grad_fn_d(params_d,static_d,audio,fake_audio)
        # Average cross the devices.
        # grads_d = jax.lax.pmean(grads_d, axis_name='num_devices')
        # loss_d = jax.lax.pmean(loss_d, axis_name='num_devices')

        updates_d, optim_g = optim_d.update(grads_d, optim_g, discriminator_model)
        discriminator_model = eqx.apply_updates(discriminator_model, updates_d)
        # Update the discriminator through gradient descent.
        # new_generator_state = generator_state.apply_gradients(
        # grads=grads_g)
        # new_discriminator_state = discriminator_state.apply_gradients(
        # grads=grads_d)
        return generator_model,optim_g,discriminator_model,optim_d,loss_g,loss_d,loss_m,loss_s
    # @partial(jax.pmap, axis_name='num_devices')         
    # def do_validate(generator: TrainState,ppg_val:jnp.ndarray,pit_val:jnp.ndarray,spk_val:jnp.ndarray,audio:jnp.ndarray):   
    #     stft = TacotronSTFT(filter_length=hp.audio.filter_length,
    #                     hop_length=hp.audio.hop_length,
    #                     win_length=hp.audio.win_length,
    #                     n_mel_channels=hp.audio.n_mel_channels,
    #                     sampling_rate=hp.audio.sampling_rate,
    #                     mel_fmin=hp.audio.mel_fmin,
    #                     mel_fmax=hp.audio.mel_fmax)   
    #     fake_audio = generator.apply_fn(
    #             {'params': generator.params},spk_val,ppg_val, pit_val,train=False, mutable=False)
    #     mel_fake = stft.mel_spectrogram(fake_audio.squeeze(1))
    #     mel_real = stft.mel_spectrogram(audio.squeeze(1))
    #     mel_loss_val = jnp.mean(optax.huber_loss(mel_fake, mel_real))

    #     #f idx == 0:
    #     spec_fake = stft.linear_spectrogram(fake_audio.squeeze(1))
    #     spec_real = stft.linear_spectrogram(audio.squeeze(1))
    #     audio = audio[0][0]
    #     fake_audio = fake_audio[0][0]
    #     spec_fake = spec_fake[0]
    #     spec_real = spec_real[0]
    #     return mel_loss_val, fake_audio, spec_fake, spec_real
    # def validate(generator):
    #     loader = tqdm.tqdm(valloader, desc='Validation loop')
       
     
    #     mel_loss = 0.0
    #     for idx, ( spk, ppg, pit, audio) in enumerate(loader): 
    #         spk = np.broadcast_to(np.asarray(spk),[8,spk.shape[1]])
    #         ppg = np.broadcast_to(np.asarray(ppg),[8,ppg.shape[1],ppg.shape[2]])
    #         pit = np.broadcast_to(np.asarray(pit),[8,pit.shape[1]])
    #         audio = np.broadcast_to(np.asarray(audio),[8,audio.shape[1],audio.shape[2]])
    #         ppg=shard(ppg)
    #         #ppg_l=shard(ppg_l)
    #         pit=shard(pit)
    #         spk=shard(spk)
    #         val_audio=shard(audio)
    #         mel_loss_val,fake_audio,spec_fake,spec_real=do_validate(generator,ppg,pit,spk,val_audio)
    #         #if idx == 0:
    #         fake_audio,spec_fake,spec_real = \
    #     jax.device_get([ fake_audio[0],spec_fake[0],spec_real[0]])
    #         writer.log_fig_audio(audio[0][0], fake_audio, idx, step)
            
    #             #res = (audio,fake_audio,spec_fake,spec_real,idx)
    #         mel_loss_val = np.mean(mel_loss_val)
    #         mel_loss += mel_loss_val
    #     mel_loss = mel_loss / len(valloader.dataset)
    #     mel_loss = np.asarray(mel_loss)
    #     writer.log_validation(mel_loss, step)
    #     #(audio,fake_audio,spec_fake,spec_real,idx) = res
        

    key = jax.random.PRNGKey(seed=hp.train.seed)
    key_generator, key_discriminator, key = jax.random.split(key, 3)
    #key_generator = shard_prng_key(key_generator)
    #key_discriminator = shard_prng_key(key_discriminator)
    
    
  
    init_epoch = 1
    step = 0
   


    if rank == 0:
        pth_dir = os.path.join(hp.log.pth_dir, args.name)
        log_dir = os.path.join(hp.log.log_dir, args.name)
        os.makedirs(pth_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, '%s-%d.log' % (args.name, time.time()))),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger()
        writer = MyWriter(hp, log_dir)
        trainloader = create_dataloader(hp, True)
        valloader = create_dataloader(hp, False)

    optim_d = optax.adamw(1e-3)
    discriminator_model = Discriminator(hp,key=key_discriminator)
    optim_g = optax.adamw(1e-3)
    generator_model = Generator(hp,key=key_generator)
    # options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    # checkpoint_manager = orbax.checkpoint.CheckpointManager(
    #     'chkpt/lora-svc/', orbax_checkpointer, options)
    # if checkpoint_manager.latest_step() is not None:
    #     target = {'model_g': generator_state, 'model_d': discriminator_state}
    #     step = checkpoint_manager.latest_step()  # step = 4
    #     states=checkpoint_manager.restore(step,items=target)
    #     discriminator_state=flax.jax_utils.replicate(states['model_d'])
    #     generator_state=flax.jax_utils.replicate(states['model_g'])
    # else:
    #     discriminator_state=flax.jax_utils.replicate(discriminator_state)
    #     generator_state=flax.jax_utils.replicate(generator_state)
    num_devices = len(jax.devices())
    devices = mesh_utils.create_device_mesh((num_devices, 1))
    shard = sharding.PositionalSharding(devices)
    for epoch in itertools.count(init_epoch+1):

        loader = tqdm.tqdm(trainloader, desc='Loading train data')

        for spk, ppg, pit, audio in loader:
            spk=jnp.asarray(spk)
            ppg=jnp.asarray(ppg)
            pit=jnp.asarray(pit)
            audio=jnp.asarray(audio)
            # print(spk.shape)
            # print(ppg.shape)
            # print(pit.shape)
            # print(audio.shape)
            print(num_devices)
            ppg,pit,spk,audio = jax.device_put((ppg,pit,spk,audio), shard)
            generator_model,optim_g,discriminator_model,optim_d,loss_g,loss_d,loss_m,loss_s=combine_step(generator_model,optim_g, discriminator_model,optim_d,ppg,pit,spk,audio)

            step += 1

            # loss_g,loss_d,loss_s,loss_m, = \
            # jax.device_get([loss_g[0], loss_d[0],loss_s[0],loss_m[0]])
            if rank == 0 and step % hp.log.info_interval == 0:
                # writer.log_training(
                #     loss_g, loss_d, loss_m, loss_s,step)
                logger.info("g %.04f m %.04f s %.04f d %.04f | step %d" % (
                    loss_g, loss_m, loss_s, loss_d, step))
        # if rank == 0 and epoch % hp.log.eval_interval == 0:
        #     validate(generator_state)
        # if rank == 0 and epoch % hp.log.save_interval == 0:
        #     generator_state_s = flax.jax_utils.unreplicate(generator_state)
        #     discriminator_state_s = flax.jax_utils.unreplicate(discriminator_state)
        #     ckpt = {'model_g': generator_state_s, 'model_d': discriminator_state_s}
        #     save_args = orbax_utils.save_args_from_target(ckpt)
        #     checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})

