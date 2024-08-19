import time

import jax
import jax.numpy as jnp

from jimgw.jim import Jim
from jimgw.prior import CombinePrior, UniformPrior, CosinePrior, SinePrior, PowerLawPrior
from jimgw.single_event.detector import H1, L1
from jimgw.single_event.likelihood import TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomPv2
from jimgw.transforms import BoundToUnbound
from jimgw.single_event.transforms import MassRatioToSymmetricMassRatioTransform, SpinToCartesianSpinTransform, ComponentMassesToChirpMassMassRatioTransform, SkyFrameToDetectorFrameSkyPositionTransform
from jimgw.single_event.utils import Mc_q_to_m1_m2
from flowMC.strategy.optimization import optimization_Adam

jax.config.update("jax_enable_x64", True)

###########################################
########## First we grab data #############
###########################################

total_time_start = time.time()

# first, fetch a 4s segment centered on GW150914
gps = 1126259462.4
duration = 4
post_trigger_duration = 2
start_pad = duration - post_trigger_duration
end_pad = post_trigger_duration
fmin = 20.0
fmax = 1024.0

ifos = [H1, L1]

f_ref = 20.0

for ifo in ifos:
    ifo.load_data(gps, start_pad, end_pad, fmin, fmax, psd_pad=16, tukey_alpha=0.2)

M_c_min, M_c_max = 10.0, 80.0
q_min, q_max = 0.125, 1.0
m_1_prior = UniformPrior(Mc_q_to_m1_m2(M_c_min, q_max)[0], Mc_q_to_m1_m2(M_c_max, q_min)[0], parameter_names=["m_1"])
m_2_prior = UniformPrior(Mc_q_to_m1_m2(M_c_min, q_min)[1], Mc_q_to_m1_m2(M_c_max, q_max)[1], parameter_names=["m_2"])
theta_jn_prior = SinePrior(parameter_names=["theta_jn"])
phi_jl_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phi_jl"])
theta_1_prior = SinePrior(parameter_names=["theta_1"])
theta_2_prior = SinePrior(parameter_names=["theta_2"])
phi_12_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phi_12"])
a_1_prior = UniformPrior(0.0, 1.0, parameter_names=["a_1"])
a_2_prior = UniformPrior(0.0, 1.0, parameter_names=["a_2"])
dL_prior = PowerLawPrior(10.0, 2000.0, 2.0, parameter_names=["d_L"])
t_c_prior = UniformPrior(-0.05, 0.05, parameter_names=["t_c"])
phase_c_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["phase_c"])
psi_prior = UniformPrior(0.0, jnp.pi, parameter_names=["psi"])
ra_prior = UniformPrior(0.0, 2 * jnp.pi, parameter_names=["ra"])
dec_prior = CosinePrior(parameter_names=["dec"])

prior = CombinePrior(
    [
        m_1_prior,
        m_2_prior,
        theta_jn_prior,
        phi_jl_prior,
        theta_1_prior,
        theta_2_prior,
        phi_12_prior,
        a_1_prior,
        a_2_prior,
        dL_prior,
        t_c_prior,
        phase_c_prior,
        psi_prior,
        ra_prior,
        dec_prior,
    ]
)

sample_transforms = [
    ComponentMassesToChirpMassMassRatioTransform(name_mapping=[["m_1", "m_2"], ["M_c", "q"]]),
    BoundToUnbound(name_mapping = [["M_c"], ["M_c_unbounded"]], original_lower_bound=10.0, original_upper_bound=80.0),
    BoundToUnbound(name_mapping = [["q"], ["q_unbounded"]], original_lower_bound=0.125, original_upper_bound=1.),
    BoundToUnbound(name_mapping = [["theta_jn"], ["theta_jn_unbounded"]] , original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = [["phi_jl"], ["phi_jl_unbounded"]] , original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    BoundToUnbound(name_mapping = [["theta_1"], ["theta_1_unbounded"]] , original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = [["theta_2"], ["theta_2_unbounded"]] , original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = [["phi_12"], ["phi_12_unbounded"]] , original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    BoundToUnbound(name_mapping = [["a_1"], ["a_1_unbounded"]] , original_lower_bound=0.0, original_upper_bound=1.0),
    BoundToUnbound(name_mapping = [["a_2"], ["a_2_unbounded"]] , original_lower_bound=0.0, original_upper_bound=1.0),
    BoundToUnbound(name_mapping = [["d_L"], ["d_L_unbounded"]] , original_lower_bound=10.0, original_upper_bound=2000.0),
    BoundToUnbound(name_mapping = [["t_c"], ["t_c_unbounded"]] , original_lower_bound=-0.05, original_upper_bound=0.05),
    BoundToUnbound(name_mapping = [["phase_c"], ["phase_c_unbounded"]] , original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
    BoundToUnbound(name_mapping = [["psi"], ["psi_unbounded"]], original_lower_bound=0.0, original_upper_bound=jnp.pi),
    SkyFrameToDetectorFrameSkyPositionTransform(name_mapping = [["ra", "dec"], ["zenith", "azimuth"]], gps_time=gps, ifos=ifos),
    BoundToUnbound(name_mapping = [["zenith"], ["zenith_unbounded"]], original_lower_bound=0.0, original_upper_bound=jnp.pi),
    BoundToUnbound(name_mapping = [["azimuth"], ["azimuth_unbounded"]], original_lower_bound=0.0, original_upper_bound=2 * jnp.pi),
]

likelihood_transforms = [
    ComponentMassesToChirpMassMassRatioTransform(name_mapping=[["m_1", "m_2"], ["M_c", "q"]]),
    SpinToCartesianSpinTransform(name_mapping=[["theta_jn", "phi_jl", "theta_1", "theta_2", "phi_12", "a_1", "a_2"], ["iota", "s1_x", "s1_y", "s1_z", "s2_x", "s2_y", "s2_z"]], freq_ref=f_ref),
    MassRatioToSymmetricMassRatioTransform(name_mapping=[["q"], ["eta"]]),
]

likelihood = TransientLikelihoodFD(
    ifos,
    waveform=RippleIMRPhenomPv2(f_ref=f_ref),
    trigger_time=gps,
    duration=4,
    post_trigger_duration=2,
)


mass_matrix = jnp.eye(15)
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[9, 9].set(1e-3)
local_sampler_arg = {"step_size": mass_matrix * 1e-3}

Adam_optimizer = optimization_Adam(n_steps=3000, learning_rate=0.01, noise_level=1)

n_epochs = 40
n_loop_training = 100
learning_rate = 1e-4


jim = Jim(
    likelihood,
    prior,
    sample_transforms=sample_transforms,
    likelihood_transforms=likelihood_transforms,
    n_loop_training=n_loop_training,
    n_loop_production=20,
    n_local_steps=10,
    n_global_steps=1000,
    n_chains=500,
    n_epochs=n_epochs,
    learning_rate=learning_rate,
    n_max_examples=30000,
    n_flow_samples=100000,
    momentum=0.9,
    batch_size=30000,
    use_global=True,
    train_thinning=1,
    output_thinning=10,
    local_sampler_arg=local_sampler_arg,
    strategies=[Adam_optimizer, "default"],
)

jim.sample(jax.random.PRNGKey(42))
jim.get_samples()
jim.print_summary()

###########################################
########## Visualize the Data #############
###########################################
import corner
import matplotlib.pyplot as plt
import numpy as np

production_summary = jim.sampler.get_sampler_state(training=False)
production_chain = production_summary["chains"].reshape(-1, len(jim.parameter_names)).T
if jim.sample_transforms:
    transformed_chain = jim.add_name(production_chain)
    for transform in reversed(jim.sample_transforms):
        transformed_chain = transform.backward(transformed_chain)
result = transformed_chain
labels = list(transformed_chain.keys())

samples = np.array(list(result.values())).reshape(int(len(labels)), -1) # flatten the array
transposed_array = samples.T # transpose the array
figure = corner.corner(transposed_array, labels=labels, plot_datapoints=False, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='g', use_math_text=True)
plt.savefig("GW1500914_Pv2.jpeg")

###########################################
############# Save the Run ################
###########################################
# import pickle
# pickle.dump(result, open("GW150914_pv2.pkl", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
