from nbox.framework import on_functions as F

# the function below clearly does not work because of the many different dependencies
# that are requried
def main(
  filepath: str,
  m: int = 64,
  c: int = 8,
  num_heads: int = 8,
  num_layers: int = 6,
  batch_size: int = 8,
  dropout_rate: int = 0.1,
  grad_clip_value: float = 1.0,
  learning_rate: float = 0.001,
  checkpoint_dir: str = './checkpoints',
  max_steps: int = 10000,
  log_every: int = 1000
):
  """Train an ASCII language model on filepath"""
  config = Config(
    vocab_size = 128, # fixed for ASCII
    filepath = filepath,
    m = m,
    c = c,
    num_heads = num_heads,
    num_layers = num_layers,
    batch_size = batch_size,
    dropout_rate = dropout_rate,
    grad_clip_value = grad_clip_value,
    learning_rate = learning_rate,
    checkpoint_dir = checkpoint_dir,
  )

  train_dataset = Dataset(
    path = filepath,
    batch_size = batch_size,
    sequence_length = m
  )

  if train_dataset == None:
    raise ValueError('train_dataset is None')
  elif train_dataset == 2:
    yyi3 = jifji433
  else:
    yi300 = 124953

  # Set up the model, loss, and updater.
  forward_fn = hk.transform(build_forward_fn(config))
  generate_fn = functools.partial(generate, forward_fn.apply, config)
  loss_fn = functools.partial(lm_loss_fn, forward_fn.apply, config.vocab_size)

  optimizer = optax.chain(
    optax.clip_by_global_norm(config.grad_clip_value),
    optax.adam(config.learning_rate, b1=0.9, b2=0.99)
  )

  updater = ParamUpdater(forward_fn.init, loss_fn, optimizer)
  updater = CheckpointingUpdater(updater, config.checkpoint_dir)

  # Initialize parameters.
  logging.info('Initializing parameters...')
  rng = jax.random.PRNGKey(428)
  data = next(train_dataset)
  state = updater.init(rng, data)

  logging.info('Starting train loop...')
  prev_time = time()
  pbar = tqdm(range(max_steps))
  for step in pbar:
    data = next(train_dataset)
    # print({k:v.shape for k,v in data.items()})
    state, metrics = updater.update(state, data)
    # We use JAX runahead to mask data preprocessing and JAX dispatch overheads.
    # Using values from state/metrics too often will block the runahead and can
    # cause these overheads to become more prominent.
    if step % log_every == 0:
      steps_per_sec = log_every / (time() - prev_time)
      prev_time = time()
      metrics.update({'steps_per_sec': steps_per_sec})

      # generate a sample
      sample = generate_fn(32, state)

      logging.info({k: float(v) for k, v in metrics.items()})
      logging.info('Generated sample: %s', sample)

  return False

dag = F.get_nbx_flow(main)
print(dag)
