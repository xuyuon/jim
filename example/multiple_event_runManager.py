import jax
import jax.numpy as jnp

from jimgw.single_event.runManager import MultipleEventRunManager
from jimgw.single_event.utils import Mc_q_to_m1_m2

jax.config.update("jax_enable_x64", True)

mass_matrix = jnp.eye(11)
mass_matrix = mass_matrix.at[1, 1].set(1e-3)
mass_matrix = mass_matrix.at[5, 5].set(1e-3)
mass_matrix = mass_matrix * 3e-3
local_sampler_arg = {"step_size": mass_matrix}

run_manager = MultipleEventRunManager()


M_c_min, M_c_max = 5.0, 100.0
q_min, q_max = 0.125, 1.0

priors={
    "m_1": {"name": "UniformPrior", "xmin": Mc_q_to_m1_m2(M_c_min, q_max)[0], "xmax": Mc_q_to_m1_m2(M_c_max, q_min)[0]},
    "m_2": {"name": "UniformPrior", "xmin": Mc_q_to_m1_m2(M_c_min, q_max)[1], "xmax": Mc_q_to_m1_m2(M_c_max, q_min)[1]},
    "s1_z": {"name": "UniformPrior", "xmin": -1.0, "xmax": 1.0},
    "s2_z": {"name": "UniformPrior", "xmin": -1.0, "xmax": 1.0},
    "d_L": {"name": "PowerLawPrior", "xmin": 1.0, "xmax": 2000.0, "alpha": 2.0},
    "t_c": {"name": "UniformPrior", "xmin": -0.05, "xmax": 0.05},
    "phase_c": {"name": "UniformPrior", "xmin": 0.0, "xmax": 2 * jnp.pi},
    "iota": {"name": "SinePrior"},
    "psi": {"name": "UniformPrior", "xmin": 0.0, "xmax": jnp.pi},
    "ra": {"name": "UniformPrior", "xmin": 0.0, "xmax": 2 * jnp.pi},
    "dec": {"name": "CosinePrior"},
}
likelihood_parameters={"name": "HeterodynedTransientLikelihoodFD"}
waveform_parameters={"name": "RippleIMRPhenomD", "f_ref": 20.0},
sample_transforms=[
    {"name": "ComponentMassesToChirpMassMassRatioTransform", "name_mapping": [["m_1", "m_2"], ["M_c", "q"]]},
    {"name": "BoundToUnbound", "name_mapping": [["M_c"], ["M_c_unbounded"]], "original_lower_bound": 10.0, "original_upper_bound": 80.0,},
    {"name": "BoundToUnbound", "name_mapping": [["q"], ["q_unbounded"]], "original_lower_bound": 0.0, "original_upper_bound": 1.0,},
    {"name": "BoundToUnbound", "name_mapping": [["s1_z"], ["s1_z_unbounded"]], "original_lower_bound": -1.0, "original_upper_bound": 1.0,},
    {"name": "BoundToUnbound", "name_mapping": [["s2_z"], ["s2_z_unbounded"]], "original_lower_bound": -1.0, "original_upper_bound": 1.0,},
    {"name": "BoundToUnbound", "name_mapping": [["d_L"], ["d_L_unbounded"]], "original_lower_bound": 1.0, "original_upper_bound": 2000.0,},
    {"name": "BoundToUnbound", "name_mapping": [["t_c"], ["t_c_unbounded"]], "original_lower_bound": -0.05, "original_upper_bound": 0.05,},
    {"name": "BoundToUnbound", "name_mapping": [["phase_c"], ["phase_c_unbounded"]], "original_lower_bound": 0.0, "original_upper_bound": 2 * jnp.pi,},
    {"name": "BoundToUnbound", "name_mapping": [["iota"], ["iota_unbounded"]], "original_lower_bound": 0.0, "original_upper_bound": jnp.pi,},
    {"name": "BoundToUnbound", "name_mapping": [["psi"], ["psi_unbounded"]], "original_lower_bound": 0.0, "original_upper_bound": jnp.pi,},
    {"name": "BoundToUnbound", "name_mapping": [["ra"], ["ra_unbounded"]], "original_lower_bound": 0.0, "original_upper_bound": 2 * jnp.pi,},
    {"name": "BoundToUnbound", "name_mapping": [["dec"], ["dec_unbounded"]], "original_lower_bound": 0.0, "original_upper_bound": jnp.pi,},
]
likelihood_transforms=[
    {"name": "ComponentMassesToChirpMassSymmetricMassRatioTransform", "name_mapping": [["m_1", "m_2"], ["M_c", "eta"]]},
]
jim_parameters={
    "n_loop_training": 100,
    "n_loop_production": 20,
    "n_local_steps": 10,
    "n_global_steps": 1000,
    "n_chains": 500,
    "n_epochs": 30,
    "learning_rate": 1e-4,
    "n_max_examples": 30000,
    "momentum": 0.9,
    "batch_size": 30000,
    "use_global": True,
    "train_thinning": 1,
    "output_thinning": 10,
    "local_sampler_arg": local_sampler_arg,
}

run_manager.run_multiple_events("event_config.yaml", priors, waveform_parameters, likelihood_parameters, sample_transforms, likelihood_transforms, jim_parameters)