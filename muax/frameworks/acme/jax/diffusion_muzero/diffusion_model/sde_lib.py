"""Abstract SDE classes"""
import abc
import jax.numpy as jnp
import jax
import numpy as np

from muax.frameworks.acme.jax.diffusion_muzero.diffusion_model.utils import batch_mul


class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, rng, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a JAX tensor.
      t: a JAX float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * jnp.sqrt(dt)
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score = score_fn(x, t)
        drift = drift - batch_mul(diffusion ** 2, score * (0.5 if self.probability_flow else 1.))
        # Set the diffusion function to zero for ODEs.
        diffusion = jnp.zeros_like(diffusion) if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        rev_f = f - batch_mul(G ** 2, score_fn(x, t) * (0.5 if self.probability_flow else 1.))
        rev_G = jnp.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()
  

class RectifiedFlow(SDE):
    def __init__(self, N, init_type='gaussian', noise_scale=1.0, reflow_flag=False,
                 reflow_t_schedule='uniform', reflow_loss='l2', use_ode_solver='rk45',
                 sigma_var=0.0, ode_tol=1e-5, sample_N=None):
        super().__init__(N)
        self.init_type = init_type
        self.noise_scale = noise_scale
        self.sigma_var = sigma_var
        self.use_ode_solver = use_ode_solver
        self.ode_tol = ode_tol
        self.sample_N = sample_N if sample_N is not None else N
        self.sigma_t = lambda t: (1.0 - t) * sigma_var
        print('Init. Distribution Variance:', self.noise_scale)
        print('SDE Sampler Variance:', sigma_var)
        print('ODE Tolerance:', self.ode_tol)
        print('Number of sampling steps:', self.sample_N)

        self.reflow_flag = reflow_flag
        if self.reflow_flag:
            self.reflow_t_schedule = reflow_t_schedule
            self.reflow_loss = reflow_loss
            if 'lpips' in reflow_loss:
                raise NotImplementedError("LPIPS loss is not implemented for JAX version.")

    @property
    def T(self):
        return 1.0

    def sde(self, x, t):
        """Define the drift and diffusion functions for the SDE."""
        drift = jnp.zeros_like(x)
        diffusion = self.sigma_t(t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        """Compute the mean and standard deviation of p_t(x)."""
        mean = x
        std = self.sigma_t(t)
        return mean, std

    def prior_sampling(self, rng, shape):
        """Sample from the prior distribution."""
        if self.init_type == 'gaussian':
            return jax.random.normal(rng, shape) * self.noise_scale
        else:
            raise NotImplementedError("Initialization type not implemented.")

    def prior_logp(self, z):
        """Compute log-density of the prior distribution."""
        if self.init_type == 'gaussian':
            shape = z.shape
            N = np.prod(shape[1:])
            logps = -N / 2.0 * jnp.log(2 * np.pi * self.noise_scale ** 2) - \
                    jnp.sum(z ** 2, axis=tuple(range(1, len(shape)))) / (2 * self.noise_scale ** 2)
            return logps
        else:
            raise NotImplementedError("Initialization type not implemented.")

    def euler_ode(self, init_input, model, reverse=False):
        """Run ODE solver using Euler method."""
        dt = 1.0 / self.sample_N
        model_fn = mutils.get_model_fn(model, train=False)
        shape = init_input.shape

        if reverse:
            time_grid = jnp.linspace(self.T, 1e-3, self.sample_N)
        else:
            time_grid = jnp.linspace(1e-3, self.T, self.sample_N)

        def step(x, t):
            vec_t = jnp.ones(shape[0]) * t
            pred = model_fn(x, vec_t * 999)
            x = x + pred * dt
            return x

        def body_fn(i, x):
            t = time_grid[i]
            x = step(x, t)
            return x

        x = init_input
        x = jax.lax.fori_loop(0, self.sample_N, body_fn, x)
        return x

    def ode(self, init_input, model, reverse=False):
        """Run ODE solver for reflow. `init_input` can be from `pi_0` or `pi_1`."""
        model_fn = mutils.get_model_fn(model, train=False)
        shape = init_input.shape

        def ode_func(x, t):
            vec_t = jnp.ones(shape[0]) * t
            drift = model_fn(x, vec_t * 999)
            return drift

        if reverse:
            t_span = jnp.array([self.T, 1e-3])
        else:
            t_span = jnp.array([1e-3, self.T])

        solution = ode.odeint(
            ode_func,
            init_input,
            t_span,
            rtol=self.ode_tol,
            atol=self.ode_tol,
            mxstep=1000
        )
        x = solution[-1]
        return x

    def get_z0(self, batch_shape, rng):
        """Get initial sample z0 from the prior distribution."""
        if self.init_type == 'gaussian':
            return jax.random.normal(rng, batch_shape) * self.noise_scale
        else:
            raise NotImplementedError("Initialization type not implemented.")