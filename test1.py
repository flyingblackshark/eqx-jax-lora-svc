import equinox as eqx
import jax
import jax.experimental.mesh_utils as mesh_utils
import jax.numpy as jnp
import jax.random as jr
import jax.sharding as sharding
import numpy as np
import optax  # https://github.com/deepmind/optax

# Hyperparameters
dataset_size = 64
channel_size = 4
hidden_size = 32
depth = 1
learning_rate = 3e-4
num_steps = 10
batch_size = 16  # must be a multiple of our number of devices.

# Generate some synthetic data
xs = np.random.normal(size=(dataset_size, channel_size))
ys = np.sin(xs)

model = eqx.nn.MLP(channel_size, channel_size, hidden_size, depth, key=jr.PRNGKey(6789))
optim = optax.adam(learning_rate)
opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))


def compute_loss(model, x, y):
    pred_y = jax.vmap(model)(x)
    return jnp.mean((y - pred_y) ** 2)


@eqx.filter_jit
def make_step(model, opt_state, x, y):
    #grads = eqx.filter_grad(compute_loss)(model, x, y)
    value, grads = eqx.filter_value_and_grad(compute_loss)(model,x,y)
    jax.debug.print("{}",value)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, opt_state
def dataloader(arrays, batch_size):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = np.arange(dataset_size)
    while True:
        perm = np.random.permutation(indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size

num_devices = len(jax.devices())
devices = mesh_utils.create_device_mesh((num_devices, 1))
shard = sharding.PositionalSharding(devices)

for step, (x, y) in zip(range(num_steps), dataloader((xs, ys), batch_size)):
    x, y = jax.device_put((x, y), shard)
    model, opt_state = make_step(model, opt_state, x, y)