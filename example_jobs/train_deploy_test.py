#!/usr/bin/env python3

# this is a sample code that trains a model on cifar10 dataset
# deploys the model using nbox and can test the model on test dataset

from functools import partial
import os
import json
import torch
import random
from uuid import uuid1
from tqdm.auto import trange
from tempfile import gettempdir
from torchvision import datasets

from gperc import BinaryConfig, Perceiver
from gperc.trainer import Trainer
from gperc.arrow import ArrowConfig, ArrowConsumer

from nbox import Operator, Model
from nbox.utils import Pool
from nbox.operators import NboxInstanceStartOperator

class Downloader(Operator):
  def __init__(self):
    super().__init__()

  def forward(self, target_dir = gettempdir() + "/nbx_sample"):
    # create dir if needed
    os.makedirs(target_dir, exist_ok = True)
    ds_train = datasets.CIFAR10(target_dir, download = True, train = True)
    ds_test = datasets.CIFAR10(target_dir, download = True, train = False)

    # create labels
    labels = [ "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    class_to_id = {x:i for i,x in enumerate(labels)}

    # create a target files
    target_dir = target_dir + "/cifar10"
    train_json = target_dir + "/train.json"
    test_json = target_dir + "/test.json"
    if not os.path.exists(target_dir) or not os.path.exists(train_json) or not os.path.exists(test_json):
      os.makedirs(target_dir, exist_ok=True)

      # first create the training dataset
      print("Creating dataset")
      truth = {}
      for _, (x, l) in zip(trange(len(ds_train)), ds_train):
        fp = os.path.join(target_dir, str(uuid1()) + random.choice([".png", ".jpg", ".tif"]))
        truth[fp] = labels[l]
        x.save(fp)
        
      with open(train_json, "w") as f:
        f.write(json.dumps(truth))

      # now create the test dataset
      truth_test = {}
      for _, (x, l) in zip(trange(len(ds_test)), ds_test):
        fp = os.path.join(target_dir, str(uuid1()) + random.choice([".png", ".jpg", ".tif"]))
        truth_test[fp] = labels[l]
        x.save(fp)

      with open(test_json, "w") as f:
        f.write(json.dumps(truth_test))
    
    with open(train_json) as f:
      train_data = json.loads(f.read())
    with open(test_json) as f:
      test_data = json.loads(f.read())
    
    return train_data, test_data, class_to_id

class TrainOperator(Operator):
  def __init__(self):
    super().__init__()
    self.best_model_fpath = None
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  def get_trainer(
    self,
    train_json,
    test_json,
    class_to_id,
    batch_size_train,
    batch_size_test
  ):
    # load the dataset
    data = ArrowConfig(
      n_bytes = 1,
      seqlen = "auto",
      class_to_id = class_to_id,
      callbacks = [lambda x: x.create_batches(batch_size_train,),]
    )._train_(
      fps = test_json,
    )._test_(
      fps = test_json,
    ).split(ArrowConsumer)
    data.train.create_batches(batch_size_train)
    data.test.create_batches(batch_size_test)

    # create a model and optimiser
    model = Perceiver(
      BinaryConfig(
        seqlen = data.train.seqlen,
        vocab_size = data.train.vocab_size,
        latent_dim = 32,
        latent_frac = 0.01,
        n_classes = len(class_to_id),
        ffw_ratio=1.0,
        num_heads = 2,
        num_layers = 1,
        decoder_reduction = "mean"
      )
    )
    optim = model.get_optimizer(
      "Adam",
      lr = 0.001,
      betas = (0.9, 0.999),
      eps = 1e-8,
      weight_decay = 0.0
    )

    trainer = Trainer(
      model = model,
      train_data = data.train,
      test_data = data.test,
      optim = optim,
      save_folder = "./gperc_model",
      save_every = 10,
      gpu = 1,
      logger_client = None,
    )

    return trainer

  def forward(
    self,
    n_steps,
    test_every,
    train_json,
    test_json,
    class_to_id,
    batch_size_train,
    batch_size_test,
    deployment_id = None,
    model_name = None,
  ):
    # get all the required things
    trainer = self.get_trainer(
      train_json, test_json, class_to_id, batch_size_train, batch_size_test
    )
    best_model_fpath = trainer.train(n_steps, test_every)
    sample_data = trainer.test_data[0]
    sample_data.pop("class")
    trainer.load_from_tar(best_model_fpath)
    
    # create an nbox model and deploy
    url, key = Model(
      trainer.model.eval(),
      category = {
        "input_array": "text",
        "attention_mask": "text"
      },
      cache_dir="./gperc_model",
    ).deploy(
      input_object = {k:v.numpy() for k,v in sample_data.items()},
      model_name = model_name,
      wait_for_deployment = True,
      runtime = "onnx",
      deployment_type = "nbox",
      deployment_id = deployment_id,
    )

    return url, key

class TestDeploy(Operator):
  def __init__(self, workers):
    super().__init__()
    self.workers = workers
    self.pool = Pool("thread", workers)

  def forward(self, url, key, n_hits = 100):
    def __test_deploy(url, key, n_hits):
      model = Model(url, key)
      for _ in range(n_hits):
        out = model(
          "https://picsum.photos/200"
        )
    
    partial_test_deploy = partial(__test_deploy, url, key)
    self.pool(
      partial_test_deploy,
      *[
        (n_hits//self.workers,) for _ in range(self.workers)
      ]
    )

class TrainTestDeploy(Operator):
  def __init__(self):
    super().__init__()
    self.downloader = Downloader()
    self.trainer = TrainOperator()
    self.deploy_tester = TestDeploy(workers = 2)

  def forward(self, n_steps = 100, test_every = 10, batch_size = 32):
    from uuid import uuid4
    train_json, test_json, class_to_id = self.downloader()
    url, api_key = self.trainer(
      n_steps = n_steps,
      test_every = test_every,
      train_json = train_json,
      test_json = test_json,
      class_to_id = class_to_id,
      batch_size_train = batch_size,
      batch_size_test = batch_size * 4,
      model_name = str(uuid4()).replace("-", "_")[:43]
    )
    self.deploy_tester(url, api_key, n_hits = 20)

# local test
# job = TrainTestDeploy()
# job(n_steps = 10, test_every = 1, batch_size = 5)
