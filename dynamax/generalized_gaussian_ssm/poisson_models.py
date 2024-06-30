from jaxtyping import Array, Float
import tensorflow_probability.substrates.jax as tfp
from typing import NamedTuple, Optional, Union, Callable

tfd = tfp.distributions
tfb = tfp.bijectors
MVN = tfd.MultivariateNormalFullCovariance

from dynamax.ssm import SSM
from dynamax.nonlinear_gaussian_ssm.models import FnStateToState, FnStateAndInputToState
from dynamax.nonlinear_gaussian_ssm.models import FnStateToEmission, FnStateAndInputToEmission


from dynamax.linear_gaussian_ssm.inference import ParamsLGSSMInitial,ParamsLGSSMDynamics
from dynamax.linear_gaussian_ssm.models import LinearGaussianSSM
import jax.random as jr
from jax.random import PRNGKey
from typing import Tuple
import jax.numpy as jnp
from dynamax.utils.bijectors import RealToPSDBijector
from dynamax.generalized_gaussian_ssm.models import ParamsGGSSM
from dynamax.linear_gaussian_ssm.models import ParamsLGSSM, LinearGaussianSSM, PosteriorGSSMFiltered
from dynamax.parameters import ParameterProperties, ParameterSet
from dynamax.linear_gaussian_ssm.inference import preprocess_params_and_inputs, _get_params, _get_one_param, tree_map, lax

FnStateToEmission2 = Callable[[Float[Array, "state_dim"]], Float[Array, "emission_dim emission_dim"]]
FnStateAndInputToEmission2 = Callable[[Float[Array, "state_dim"], Float[Array, "input_dim"]], Float[Array, "emission_dim emission_dim"]]

# emission distribution takes a mean vector and covariance matrix and returns a distribution
EmissionDistFn = Callable[ [Float[Array, "state_dim"], Float[Array, "state_dim state_dim"]], tfd.Distribution]


def lgssm_joint_sample(
        params: ParamsLGSSM,
        key: PRNGKey,
        num_timesteps: int,
        inputs: Optional[Float[Array, "num_timesteps input_dim"]] = None
) -> Tuple[Float[Array, "num_timesteps state_dim"],
Float[Array, "num_timesteps emission_dim"]]:
    r"""Sample from the joint distribution to produce state and emission trajectories.

    Args:
        params: model parameters
        inputs: optional array of inputs.

    Returns:
        latent states and emissions

    """
    params, inputs = preprocess_params_and_inputs(params, num_timesteps, inputs)

    def _sample_transition(key, F, B, b, Q, x_tm1, u):
        mean = F @ x_tm1 + B @ u + b
        return MVN(mean, Q).sample(seed=key)

    def _sample_emission(key, H, D, d, R, x, u):
        mean = H @ x + D @ u + d
        R = jnp.diag(R) if R.ndim == 1 else R
        return tfd.Poisson(log_rate=mean).sample(seed=key) #MVN(mean, R).sample(seed=key)

    def _sample_initial(key, params, inputs):
        key1, key2 = jr.split(key)

        initial_state = MVN(params.initial.mean, params.initial.cov).sample(seed=key1)

        H0, D0, d0, R0 = _get_params(params, num_timesteps, 0)[4:]
        u0 = tree_map(lambda x: x[0], inputs)

        initial_emission = _sample_emission(key2, H0, D0, d0, R0, initial_state, u0)
        return initial_state, initial_emission

    def _step(prev_state, args):
        key, t, inpt = args
        key1, key2 = jr.split(key, 2)

        # Get parameters and inputs for time index t
        F, B, b, Q, H, D, d, R = _get_params(params, num_timesteps, t)

        # Sample from transition and emission distributions
        state = _sample_transition(key1, F, B, b, Q, prev_state, inpt)
        emission = _sample_emission(key2, H, D, d, R, state, inpt)

        return state, (state, emission)

    # Sample the initial state
    key1, key2 = jr.split(key)

    initial_state, initial_emission = _sample_initial(key1, params, inputs)

    # Sample the remaining emissions and states
    next_keys = jr.split(key2, num_timesteps - 1)
    next_times = jnp.arange(1, num_timesteps)
    next_inputs = tree_map(lambda x: x[1:], inputs)
    _, (next_states, next_emissions) = lax.scan(_step, initial_state, (next_keys, next_times, next_inputs))

    # Concatenate the initial state and emission with the following ones
    expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
    states = tree_map(expand_and_cat, initial_state, next_states)
    emissions = tree_map(expand_and_cat, initial_emission, next_emissions)

    return states, emissions


class ParamsGGSSMPoissonEmissions(NamedTuple):
    r"""Parameters of the Poisson emission distribution

    $$p(y_t \mid z_t, u_t) = \text{Pois}(y_t \mid H z_t + D u_t + d)$$

    The tuple doubles as a container for the ParameterProperties.

    :param weights: emission weights $H$
    :param bias: emission bias $d$
    :param input_weights: emission input weights $D$

    """
    weights: Union[ParameterProperties,
    Float[Array, "emission_dim state_dim"],
    Float[Array, "ntime emission_dim state_dim"]]

    bias: Union[ParameterProperties,
    Float[Array, "emission_dim"],
    Float[Array, "ntime emission_dim"]]

    input_weights: Union[ParameterProperties,
    Float[Array, "emission_dim input_dim"],
    Float[Array, "ntime emission_dim input_dim"]]

    cov: Union[ParameterProperties,
        Float[Array, "emission_dim emission_dim"],
        Float[Array, "ntime emission_dim emission_dim"],
        Float[Array, "emission_dim"],
        Float[Array, "ntime emission_dim"],
        Float[Array, "emission_dim_triu"]]



class ParamsLGSSMPoisson(ParamsLGSSM):
    """
    Params for LGSSM
    """
    emissions: ParamsGGSSMPoissonEmissions



class PoissonLinearGaussianSSM(LinearGaussianSSM):
    """
    Linear Gaussian SSM model with Poisson emissions.

    The model is defined as follows

    $$p(z_1) = \mathcal{N}(z_1 \mid m, S)$$
    $$p(z_t \mid z_{t-1}, u_t) = \mathcal{N}(z_t \mid F_t z_{t-1} + B_t u_t + b_t, Q_t)$$
    $$p(y_t \mid z_t) = \text{Pois}(y_t \mid H_t z_t + D_t u_t + d_t)$$

    where

    * $z_t$ is a latent state of size `state_dim`,
    * $y_t$ is an emission of size `emission_dim`
    * $u_t$ is an input of size `input_dim` (defaults to 0)
    * $F$ = dynamics (transition) matrix
    * $B$ = optional input-to-state weight matrix
    * $b$ = optional input-to-state bias vector
    * $Q$ = covariance matrix of dynamics (system) noise
    * $H$ = emission (observation) matrix
    * $D$ = optional input-to-emission weight matrix
    * $d$ = optional input-to-emission bias vector
    * $m$ = mean of initial state
    * $S$ = covariance matrix of initial state

    The parameters of the model are stored in a :class:`ParamsLGSSM`.
    You can create the parameters manually, or by calling :meth:`initialize`.

    :param state_dim: Dimensionality of latent state.
    :param emission_dim: Dimensionality of observation vector.
    :param input_dim: Dimensionality of input vector. Defaults to 0.
    :param has_dynamics_bias: Whether model contains an offset term $b$. Defaults to True.
    :param has_emissions_bias:  Whether model contains an offset term $d$. Defaults to True.

    """
    # def initialize(
    #     self,
    #     key: PRNGKey = jr.PRNGKey(0),
    #     initial_mean: Optional[Float[Array, "state_dim"]]=None,
    #     initial_covariance=None,
    #     dynamics_weights=None,
    #     dynamics_bias=None,
    #     dynamics_input_weights=None,
    #     dynamics_covariance=None,
    #     emission_weights=None,
    #     emission_bias=None,
    #     emission_input_weights=None,
    #     emission_covariance=None
    # ) -> Tuple[ParamsLGSSMPoisson, ParamsLGSSMPoisson]:
    #     r"""Initialize model parameters that are set to None, and their corresponding properties.
    #
    #     Args:
    #         key: Random number key. Defaults to jr.PRNGKey(0).
    #         initial_mean: parameter $m$. Defaults to None.
    #         initial_covariance: parameter $S$. Defaults to None.
    #         dynamics_weights: parameter $F$. Defaults to None.
    #         dynamics_bias: parameter $b$. Defaults to None.
    #         dynamics_input_weights: parameter $B$. Defaults to None.
    #         dynamics_covariance: parameter $Q$. Defaults to None.
    #         emission_weights: parameter $H$. Defaults to None.
    #         emission_bias: parameter $d$. Defaults to None.
    #         emission_input_weights: parameter $D$. Defaults to None.
    #
    #     Returns:
    #         Tuple[ParamsLGSSM, ParamsLGSSM]: parameters and their properties.
    #     """
    #
    #     # Arbitrary default values, for demo purposes.
    #     _initial_mean = jnp.zeros(self.state_dim)
    #     _initial_covariance = jnp.eye(self.state_dim)
    #     _dynamics_weights = 0.99 * jnp.eye(self.state_dim)
    #     _dynamics_input_weights = jnp.zeros((self.state_dim, self.input_dim))
    #     _dynamics_bias = jnp.zeros((self.state_dim,)) if self.has_dynamics_bias else None
    #     _dynamics_covariance = 0.1 * jnp.eye(self.state_dim)
    #     _emission_weights = jr.normal(key, (self.emission_dim, self.state_dim))
    #     _emission_input_weights = jnp.zeros((self.emission_dim, self.input_dim))
    #     _emission_bias = jnp.zeros((self.emission_dim,)) if self.has_emissions_bias else None
    #
    #
    #     # Only use the values above if the user hasn't specified their own
    #     default = lambda x, x0: x if x is not None else x0
    #
    #     # Create nested dictionary of params
    #     params = ParamsLGSSMPoisson(
    #         initial=ParamsLGSSMInitial(
    #             mean=default(initial_mean, _initial_mean),
    #             cov=default(initial_covariance, _initial_covariance)),
    #         dynamics=ParamsLGSSMDynamics(
    #             weights=default(dynamics_weights, _dynamics_weights),
    #             bias=default(dynamics_bias, _dynamics_bias),
    #             input_weights=default(dynamics_input_weights, _dynamics_input_weights),
    #             cov=default(dynamics_covariance, _dynamics_covariance)),
    #         emissions=ParamsGGSSMPoissonEmissions(
    #             weights=default(emission_weights, _emission_weights),
    #             bias=default(emission_bias, _emission_bias),
    #             input_weights=default(emission_input_weights, _emission_input_weights))
    #         )
    #
    #     # The keys of param_props must match those of params!
    #     props = ParamsLGSSMPoisson(
    #         initial=ParamsLGSSMInitial(
    #             mean=ParameterProperties(),
    #             cov=ParameterProperties(constrainer=RealToPSDBijector())),
    #         dynamics=ParamsLGSSMDynamics(
    #             weights=ParameterProperties(),
    #             bias=ParameterProperties(),
    #             input_weights=ParameterProperties(),
    #             cov=ParameterProperties(constrainer=RealToPSDBijector())),
    #         emissions=ParamsGGSSMPoissonEmissions(
    #             weights=ParameterProperties(),
    #             bias=ParameterProperties(),
    #             input_weights=ParameterProperties())
    #         )
    #     return params, props

    def sample(
        self,
        params: ParamsLGSSM,
        key: PRNGKey,
        num_timesteps: int,
        inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> PosteriorGSSMFiltered:
        return lgssm_joint_sample(params, key, num_timesteps, inputs)

    def emission_distribution(
        self,
        params: ParamsLGSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params.emissions.weights @ state + params.emissions.input_weights @ inputs
        if self.has_emissions_bias:
            mean += params.emissions.bias
        return tfd.Poisson(log_rate=mean)
        # return MVN(mean, params.emissions.cov)

