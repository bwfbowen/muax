from typing import NamedTuple, Optional, Callable
import numpy as np
import jax
from jax import numpy as jnp 
import haiku as hk
from functools import partial

from muax.utils import action2plane


class MZNetworkParams(NamedTuple):
    representation: Optional[hk.Params] = None
    prediction: Optional[hk.Params] = None
    dynamic: Optional[hk.Params] = None
    

class MZNetwork(NamedTuple):
    representation_fn: Callable
    prediction_fn: Callable
    dynamic_fn: Callable


def create_muzero_network(
    representation_module: hk.Module,
    prediction_module: hk.Module,
    dynamic_module: hk.Module,
    embedding_dim: int,
    num_actions: int,
    full_support_size: int
) -> MZNetwork:
    repr_fn = _init_representation_func(representation_module, embedding_dim)
    pred_fn = _init_prediction_func(prediction_module, num_actions, full_support_size)
    dy_fn = _init_dynamic_func(dynamic_module, embedding_dim, num_actions, full_support_size)
    return MZNetwork(repr_fn, pred_fn, dy_fn)


@jax.jit
def min_max_normalize(s):
  s_min = s.min(axis=1, keepdims=True)
  s_max = s.max(axis=1, keepdims=True)
  s_scale = s_max - s_min 
  s_scale = jnp.where(s_scale < 1e-5, s_scale + 1e-5, s_scale)
  s_normed = (s - s_min) / (s_scale)
  return s_normed


@jax.jit
def min_max_normalize2d(s):
    s_min = jnp.expand_dims(s.reshape(-1, s.shape[1]*s.shape[2], s.shape[3]).min(axis=1, keepdims=True),
                        axis=1)
    s_max = jnp.expand_dims(s.reshape(-1, s.shape[1]*s.shape[2], s.shape[3]).max(axis=1, keepdims=True),
                        axis=1)
    s_scale = s_max - s_min 
    s_scale = jnp.where(s_scale < 1e-5, s_scale + 1e-5, s_scale)
    s_normed = (s - s_min) / s_scale 
    return s_normed


class Representation(hk.Module):
  def __init__(self, embedding_dim, name='representation'):
    super().__init__(name=name)

    self.repr_func = hk.Sequential([
        hk.Linear(embedding_dim)
    ])

  def __call__(self, obs):
    s = self.repr_func(obs)
    s = min_max_normalize(s)
    return s 


class Prediction(hk.Module):
  def __init__(self, num_actions, full_support_size, name='prediction'):
    super().__init__(name=name)        
    
    self.v_func = hk.Sequential([
        hk.Linear(16), jax.nn.elu,
        hk.Linear(full_support_size)
    ])
    self.pi_func = hk.Sequential([
        hk.Linear(16), jax.nn.elu,
        hk.Linear(num_actions)
    ])
  
  def __call__(self, s):
    v = self.v_func(s)
    logits = self.pi_func(s)
    # logits = jax.nn.softmax(logits, axis=-1)
    return v, logits


class Dynamic(hk.Module):
  def __init__(self, embedding_dim, num_actions, full_support_size, name='dynamic'):
    super().__init__(name=name)
    
    self.ns_func = hk.Sequential([
        hk.Linear(16), jax.nn.elu,
        hk.Linear(embedding_dim)
    ])
    self.r_func = hk.Sequential([
        hk.Linear(16), jax.nn.elu,
        hk.Linear(full_support_size)
    ])
    self.cat_func = jax.jit(lambda s, a: 
                            jnp.concatenate([s, jax.nn.one_hot(a, num_actions)],
                                            axis=1)
                            )
  
  def __call__(self, s, a):
    sa = self.cat_func(s, a)
    r = self.r_func(sa)
    ns = self.ns_func(sa)
    ns = min_max_normalize(ns)
    return r, ns


class ResidualConvBlockV1(hk.Module):
    """A v1 residual convolutional block."""
    def __init__(self, channels: int, stride: int, use_projection: bool, name='residual_conv_block_v1'):
        super(ResidualConvBlockV1, self).__init__(name=name)
        self._use_projection = use_projection
        if use_projection:
            self._proj_conv = hk.Conv2D(channels, kernel_shape=3, stride=stride, padding='SAME', with_bias=False)
            self._proj_ln = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)
        self._conv_0 = hk.Conv2D(channels, kernel_shape=3, stride=stride, padding='SAME', with_bias=False)
        self._ln_0 = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)
        self._conv_1 = hk.Conv2D(channels, kernel_shape=3, stride=1, padding='SAME', with_bias=False)
        self._ln_1 = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)

    def __call__(self, x):
        # NOTE: Replacing BatchNorm with LayerNorm is totally fine for RL.
        #   See https://arxiv.org/pdf/2104.06294.pdf Appendix A for an example.
        shortcut = out = x

        if self._use_projection:
            shortcut = self._proj_conv(shortcut)
            shortcut = self._proj_ln(shortcut)

        out = hk.Sequential([
            self._conv_0,
            self._ln_0,
            jax.nn.relu,
            self._conv_1,
            self._ln_1,
        ])(out)

        return jax.nn.relu(shortcut + out)


class ResidualConvBlockV2(hk.Module):
    """A v2 residual convolutional block."""
    def __init__(self, channels, stride: int, use_projection: bool, name='residual_conv_block_v2'):
        super(ResidualConvBlockV2, self).__init__(name=name)
        self._use_projection = use_projection
        if use_projection:
            self._proj_conv = hk.Conv2D(channels, kernel_shape=3, stride=stride, padding='SAME', with_bias=False)
        self._conv_0 = hk.Conv2D(channels, kernel_shape=3, stride=stride, padding='SAME', with_bias=False)
        self._ln_0 = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)
        self._conv_1 = hk.Conv2D(channels, kernel_shape=3, stride=1, padding='SAME', with_bias=False)
        self._ln_1 = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)

    def __call__(self, x):
        # NOTE: Replacing BatchNorm with LayerNorm is totally fine for RL.
        #   See https://arxiv.org/pdf/2104.06294.pdf Appendix A for an example.
        shortcut = out = x
        out = self._ln_0(out)
        out = jax.nn.relu(out)
        if self._use_projection:
            shortcut = self._proj_conv(out)
        out = hk.Sequential([
            self._conv_0,
            self._ln_1,
            jax.nn.relu,
            self._conv_1,
        ])(out)
        return shortcut + out
    

class EZStateEncoder(hk.Module):
    """EfficientZero encoder architecture."""
    def __init__(self, channels, use_v2=True, name='ez_state_encoder'):
        super(EZStateEncoder, self).__init__(name=name)
        self._channels = channels
        self._use_v2 = use_v2

    def __call__(self, observations):
        ResBlock = ResidualConvBlockV2 if self._use_v2 else ResidualConvBlockV1
        torso = [
            lambda x: x / 255.,
            hk.Conv2D(self._channels // 2, kernel_shape=3, stride=2, padding='SAME', with_bias=False),
        ]
        if not self._use_v2:
            torso.extend([
                hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
                jax.nn.relu,
            ])
        torso.append(ResBlock(self._channels // 2, stride=1, use_projection=False))
        torso.append(ResBlock(self._channels, stride=2, use_projection=True))
        torso.extend([
            ResBlock(self._channels, stride=1, use_projection=False),
            hk.AvgPool(window_shape=(3, 3, 1), strides=(2, 2, 1), padding='SAME'),
            ResBlock(self._channels, stride=1, use_projection=False),
            hk.AvgPool(window_shape=(3, 3, 1), strides=(2, 2, 1), padding='SAME'),
            ResBlock(self._channels, stride=1, use_projection=False),
        ])
        return hk.Sequential(torso)(observations)
    

class EZRepresentation(hk.Module):
  def __init__(self, embedding_dim, name='representation'):
    super().__init__(name=name)

    self.repr_func = EZStateEncoder(embedding_dim)

  def __call__(self, obs):
    s = self.repr_func(obs)
    return s 


class EZPrediction(hk.Module):
  def __init__(self, num_actions, full_support_size, output_init_scale, use_v2=True, name='prediction'):
    super().__init__(name=name)      

    self.output_init = hk.initializers.VarianceScaling(scale=output_init_scale)
    self.ResBlock = ResidualConvBlockV2 if use_v2 else ResidualConvBlockV1
    self.v_func = hk.Sequential([
        hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
        jax.nn.relu,
        hk.Conv2D(16, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
        hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
        jax.nn.relu,
        hk.Flatten(-3),
        hk.Linear(32, with_bias=False),
        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
        jax.nn.relu,
        hk.Linear(full_support_size, w_init=self.output_init),
        ])

    self.pi_func = hk.Sequential([
        hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
        jax.nn.relu,
        hk.Conv2D(16, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
        hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
        jax.nn.relu,
        hk.Flatten(-3),
        hk.Linear(32, with_bias=False),
        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
        jax.nn.relu,
        hk.Linear(num_actions, w_init=self.output_init),
    ])
  
  def __call__(self, s):
    o = self.ResBlock(s.shape[-1], stride=1, use_projection=False)(s)
    v = self.v_func(o)
    logits = self.pi_func(o)
    # logits = jax.nn.softmax(logits, axis=-1)
    return v, logits


class EZDynamic(hk.Module):
  def __init__(self, embedding_dim, num_actions, full_support_size, output_init_scale, use_v2=True, name='dynamic'):
    super().__init__(name=name)

    self._use_v2 = use_v2
    self.output_init = hk.initializers.VarianceScaling(scale=output_init_scale)
    self.ResBlock = ResidualConvBlockV2 if use_v2 else ResidualConvBlockV1
    
    # self.ns_func = hk.Sequential()
    
    self.r_func = hk.Sequential([
        hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
        jax.nn.relu,
        hk.Conv2D(16, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
        hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
        jax.nn.relu,
        hk.Flatten(-3),
        hk.Linear(32, with_bias=False),
        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
        jax.nn.relu,
        hk.Linear(full_support_size, w_init=self.output_init),
    ])
    
    def cat_func(s, a):
      n, h, w, c = s.shape 
      a_broadcast = np.ones((n, h, w, 1)) * a.reshape(n,1,1,1)
      sa = jnp.concatenate([s, a_broadcast], axis=-1)
      return sa 

    self.cat_func = jax.jit(cat_func)
  
  def __call__(self, s, a):
    C = s.shape[-1]
    shortcut = s 
    if self._use_v2:
      s = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)(s)
      s = jax.nn.relu(s)
    
    sa = self.cat_func(s, a)
    out = hk.Conv2D(C, kernel_shape=3, stride=1, padding='SAME', with_bias=False)(sa)
    if self._use_v2:
        out += shortcut
    else:
        out = hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True)(out)
        out = jax.nn.relu(out + shortcut)
    out = self.ResBlock(C, stride=1, use_projection=False)(out)
    r = self.r_func(out)
    ns = out
    return r, ns
  


class ResNetRepresentation(hk.Module):
  def __init__(self, input_channels: int = 32, name='representation'):
    super().__init__(name=name)
    
    self.repr_func = hk.Sequential(
      [lambda x: x / 255.]
      + [hk.Conv2D(input_channels, kernel_shape=3, stride=2, padding='SAME', with_bias=False), jax.nn.relu] 
      + [ResidualConvBlockV1(channels=input_channels, stride=1, use_projection=True) for _ in range(2)]
      + [hk.Conv2D(input_channels * 2, kernel_shape=3, stride=2, padding='SAME', with_bias=False), jax.nn.relu]
      + [ResidualConvBlockV1(channels=input_channels * 2, stride=1, use_projection=True) for _ in range(3)]
      + [hk.AvgPool(window_shape=(3, 3, 1), strides=(2, 2, 1), padding='SAME')]
      + [ResidualConvBlockV1(channels=input_channels * 2, stride=1, use_projection=True) for _ in range(3)]
      + [hk.AvgPool(window_shape=(3, 3, 1), strides=(2, 2, 1), padding='SAME')]
    )

  def __call__(self, obs):
    s = self.repr_func(obs)
    s = min_max_normalize2d(s)
    return s 
  

class ResNetPrediction(hk.Module):
  def __init__(self,  num_actions, full_support_size, output_channels: int = 16, name='prediction'):
    super().__init__(name=name) 
    
    self.pi_func = hk.Sequential([
      hk.Conv2D(output_channels, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
      jax.nn.relu,
      hk.Flatten(),
      hk.Linear(output_channels),
      jax.nn.relu,
      hk.Linear(num_actions)
      ])
    
    self.v_func = hk.Sequential([
      hk.Conv2D(output_channels, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
      jax.nn.relu,
      hk.Conv2D(output_channels, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
      jax.nn.relu,
      hk.Flatten(),
      hk.Linear(output_channels),
      jax.nn.relu,
      hk.Linear(full_support_size)
      ])
  
  def __call__(self, s):
    v = self.v_func(s)
    logits = self.pi_func(s)
    return v, logits


class ResNetDynamic(hk.Module):
  def __init__(self, num_actions, full_support_size, output_channels: int = 64, name='dynamic'):
    super().__init__(name=name)

    self.r_func = hk.Sequential([
      hk.Conv2D(output_channels, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
      jax.nn.relu,
      hk.Conv2D(output_channels, kernel_shape=1, stride=1, padding='SAME', with_bias=False),
      jax.nn.relu,
      hk.Flatten(),
      hk.Linear(output_channels),
      jax.nn.relu,
      hk.Linear(full_support_size)
      ])

    self.ns_func = hk.Sequential(
      [hk.Conv2D(output_channels, kernel_shape=1, stride=1, padding='SAME', with_bias=False), jax.nn.relu,]
      + [ResidualConvBlockV1(channels=output_channels, stride=1, use_projection=True) for _ in range(8)]
      )

    self.cat_func = jax.jit(lambda s, a: jnp.concatenate([s, a], axis=-1))
    self.a_norm_func = jax.jit(lambda a: a / num_actions)
    
  def __call__(self, s, a):
    n, h, w, c = s.shape 
    a = self.a_norm_func(a)
    va2p = jax.vmap(partial(action2plane, shape=(h,w,1)))
    sa = self.cat_func(s, va2p(a))
    r = self.r_func(sa)
    ns = self.ns_func(sa)
    ns = min_max_normalize2d(ns)
    return r, ns  


def _init_ez_representation_func(representation_module, embedding_dim):
    def representation_func(obs):
      repr_model = representation_module(embedding_dim)
      return repr_model(obs)
    return representation_func
  
def _init_ez_prediction_func(prediction_module, num_actions, full_support_size, output_init_scale):
  def prediction_func(s):
    pred_model = prediction_module(num_actions, full_support_size, output_init_scale)
    return pred_model(s)
  return prediction_func

def _init_ez_dynamic_func(dynamic_module, embedding_dim, num_actions, full_support_size, output_init_scale):
  def dynamic_func(s, a):
    dy_model = dynamic_module(embedding_dim, num_actions, full_support_size, output_init_scale)
    return dy_model(s, a)
  return dynamic_func 


def _init_representation_func(representation_module, embedding_dim):
    def representation_func(obs):
      repr_model = representation_module(embedding_dim)
      return repr_model(obs)
    return representation_func
  
def _init_prediction_func(prediction_module, num_actions, full_support_size):
  def prediction_func(s):
    pred_model = prediction_module(num_actions, full_support_size)
    return pred_model(s)
  return prediction_func

def _init_dynamic_func(dynamic_module, embedding_dim, num_actions, full_support_size):
  def dynamic_func(s, a):
    dy_model = dynamic_module(embedding_dim, num_actions, full_support_size)
    return dy_model(s, a)
  return dynamic_func 
    

def _init_resnet_representation_func(representation_module, input_channels):
    def representation_func(obs):
      repr_model = representation_module(input_channels=input_channels)
      return repr_model(obs)
    return representation_func
  
def _init_resnet_prediction_func(prediction_module, num_actions, full_support_size, output_channels):
  def prediction_func(s):
    pred_model = prediction_module(num_actions, full_support_size, output_channels)
    return pred_model(s)
  return prediction_func

def _init_resnet_dynamic_func(dynamic_module, num_actions, full_support_size, output_channels):
  def dynamic_func(s, a):
    dy_model = dynamic_module(num_actions, full_support_size, output_channels)
    return dy_model(s, a)
  return dynamic_func 