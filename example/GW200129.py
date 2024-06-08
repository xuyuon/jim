############################## Import Module ##############################
import time
from jimgw.jim import Jim
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD, TransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomD, RippleIMRPhenomPv2
from jimgw.prior import Uniform, Composite, Sphere, Unconstrained_Uniform
import jax.numpy as jnp
import jax
import numpy as np
import corner
import matplotlib.pyplot as plt
import h5py
from gwosc.datasets import find_datasets, event_gps
from gwosc import datasets
from pathlib import Path
import os
from tap import Tap

from flowMC.strategy.optimization import optimization_Adam
output_dir = 'GW200129'




############################## Run Parameter Estimation for a Single Event ##############################
def runParameterEstimation(event):
    ############################## Fetch event data ##############################
    total_time_start = time.time()

    # first, fetch a 4s segment centered on the event
    gps = event_gps(event) # trigger time
    duration = 8 # Analysis segment duration
    post_trigger_duration = 2
    end = gps + post_trigger_duration
    start = end - duration
    fmin = 20.0
    fmax = 896.0
    roll_off = 0.4
    psd_alpha = 2 * roll_off / duration
    f_ref = 20.0
    
    detectors = []
    H1.load_data(gps, duration-post_trigger_duration, post_trigger_duration, fmin, fmax, psd_pad=4*duration, tukey_alpha=psd_alpha)
    detectors.append(H1)
    
    L1.load_data(gps, duration-post_trigger_duration, post_trigger_duration, fmin, fmax, psd_pad=4*duration, tukey_alpha=psd_alpha)
    detectors.append(L1)
    
    V1.load_data(gps, duration-post_trigger_duration, post_trigger_duration, fmin, fmax, psd_pad=4*duration, tukey_alpha=psd_alpha)
    detectors.append(V1)
    

    waveform = RippleIMRPhenomPv2(f_ref=f_ref)
    # waveform = RippleIMRPhenomD()

    
    ############################## Set up Priors ##############################
    Mc_prior = Unconstrained_Uniform(4.5, 48.9, naming=["M_c"])
    q_prior = Unconstrained_Uniform(
        0.05,
        1.0,
        naming=["q"],
        transforms={"q": ("eta", lambda params: params["q"] / (1 + params["q"]) ** 2)},
    )
    s1_prior = Sphere(naming="s1")
    s2_prior = Sphere(naming="s2")
    # s1_z_prior = Unconstrained_Uniform(-0.99, 0.99, naming=["s1_z"]) 
    # s2_z_prior = Unconstrained_Uniform(-0.99, 0.99, naming=["s2_z"])
    dL_prior = Unconstrained_Uniform(100.0, 10000.0, naming=["d_L"])
    t_c_prior = Unconstrained_Uniform(-0.1, 0.1, naming=["t_c"]) 
    phase_c_prior = Unconstrained_Uniform(0.0, 2 * jnp.pi, naming=["phase_c"])
    cos_iota_prior = Unconstrained_Uniform(
        -1.0,
        1.0,
        naming=["cos_iota"],
        transforms={
            "cos_iota": (
                "iota",
                lambda params: jnp.arccos(
                    jnp.arcsin(jnp.sin(params["cos_iota"] / 2 * jnp.pi)) * 2 / jnp.pi
                ),
            )
        },
    )
    psi_prior = Unconstrained_Uniform(0.0, jnp.pi, naming=["psi"])
    ra_prior = Unconstrained_Uniform(0.0, 2 * jnp.pi, naming=["ra"])
    sin_dec_prior = Unconstrained_Uniform(
        -1.0,
        1.0,
        naming=["sin_dec"],
        transforms={
            "sin_dec": (
                "dec",
                lambda params: jnp.arcsin(
                    jnp.arcsin(jnp.sin(params["sin_dec"] / 2 * jnp.pi)) * 2 / jnp.pi
                ),
            )
        },
    )

    prior = Composite(
        [
            Mc_prior,
            q_prior,
            s1_prior,
            s2_prior,
            dL_prior,
            t_c_prior,
            phase_c_prior,
            cos_iota_prior,
            psi_prior,
            ra_prior,
            sin_dec_prior,
        ],
    )

    bounds = jnp.array(
        [
            [4.5, 48.9], # chirp mass
            [0.05, 1.0], # mass ratio
            [0, jnp.pi],
            [0, 2 * jnp.pi],
            [0.0, 1.0],
            [0, jnp.pi],
            [0, 2 * jnp.pi],
            [0.0, 1.0],
            [100.0, 10000.0], # luminosity distance
            [-0.1, 0.1], # geocent time
            [0.0, 2 * jnp.pi], # phase
            [-1.0, 1.0], # cos iota
            [0.0, jnp.pi], # psi
            [0.0, 2 * jnp.pi], # ra
            [-1.0, 1.0], # sin dec
        ]
    )
    
    ############################## Set up Likelihood ##############################
    # likelihood = TransientLikelihoodFD([H1, L1], waveform=waveform, trigger_time=gps, duration=4, post_trigger_duration=2)
    print("check point!!!!")
    likelihood = HeterodynedTransientLikelihoodFD(detectors, prior=prior, bounds=bounds, waveform=waveform, trigger_time=gps, duration=duration, post_trigger_duration=post_trigger_duration)


    ############################## Set up Jim Sampler ##############################
    mass_matrix = jnp.eye(prior.n_dim)
    mass_matrix = mass_matrix.at[1, 1].set(1e-3)
    mass_matrix = mass_matrix.at[9, 9].set(1e-3)
    local_sampler_arg = {"step_size": mass_matrix * 3e-3}
    
    jim = Jim(
        likelihood,
        prior,
        n_loop_training=200,
        n_loop_production=10,
        n_local_steps=300,
        n_global_steps=300,
        n_chains=500,
        n_epochs=500,
        learning_rate=0.001,
        max_samples = 10000,
        momentum=0.9,
        batch_size=10000,
        use_global=True,
        keep_quantile=0.,
        train_thinning=1,
        output_thinning=30,
        local_sampler_arg=local_sampler_arg,
    )
    
    jim.sample(jax.random.PRNGKey(42))
    
    result = jim.get_samples()
    
    summary = jim.Sampler.get_sampler_state(training=True)
    
    return result, summary


############################## Plot Posterior Samples ##############################
def plotPosterior(result, event):
    # labels = ["M_c", "eta", "s_1_z", "s_2_z", "dL", "t_c", "phase_c", "iota", "psi", "ra", "dec"]
    labels = ["M_c", "eta", "s_1_i", "s_1_j", "s_1_k", "s_2_i", "s_2_j", "s_2_k", "dL", "t_c", "phase_c", "iota", "psi", "ra", "dec"]
    
    samples = np.array(list(result.values())).reshape(15, -1) # flatten the array
    transposed_array = samples.T # transpose the array
    figure = corner.corner(transposed_array, labels=labels, plot_datapoints=False, title_quantiles=[0.16, 0.5, 0.84], show_titles=True, title_fmt='g', use_math_text=True)
    mkdir(output_dir + "/posterior_plot")
    plt.savefig(output_dir + "/posterior_plot/"+event+".jpeg")

############################## Save Posterior Samples ##############################
def savePosterior(result, event):
    samples = np.array(list(result.values())).reshape(15, -1) # flatten the array
    transposed_array = samples.T # transpose the array
    mkdir(output_dir + "/posterior_samples")
    with h5py.File(output_dir + '/posterior_samples/' + event + '.h5', 'w') as f:
        f.create_dataset('posterior', data=transposed_array)
        
def plotRunAnalysis(summary, event):
    chains, log_prob, local_accs, global_accs, loss_vals = summary.values()
    rng_key = jax.random.PRNGKey(42)
    rng_key, subkey = jax.random.split(rng_key)
    
    chains = np.array(chains)
    loss_vals = np.array(loss_vals)
    log_prob = np.array(log_prob)
    
    print(np.shape(log_prob))
    print(log_prob)
    # Plot one chain to show the jump
    plt.figure(figsize=(6, 6))
    axs = [plt.subplot(2, 2, i + 1) for i in range(4)]
    plt.sca(axs[0])
    plt.title("2 chains")
    plt.plot(chains[0, :, 0], chains[0, :, 1], alpha=0.5)
    plt.plot(chains[1, :, 0], chains[1, :, 1], alpha=0.5)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.sca(axs[1])
    plt.title("NF loss")
    plt.plot(loss_vals.reshape(-1))
    plt.xlabel("iteration")

    plt.sca(axs[2])
    plt.title("Local Acceptance")
    plt.plot(local_accs.mean(0))
    plt.xlabel("iteration")

    plt.sca(axs[3])
    plt.title("Global Acceptance")
    plt.plot(global_accs.mean(0))
    plt.xlabel("iteration")
    plt.tight_layout()
    
    mkdir(output_dir + "/posterior_analysis")
    plt.savefig(output_dir + "/posterior_analysis/"+event+".jpeg")


def plotLikelihood(summary, event):
    chains, log_prob, local_accs, global_accs, loss_vals = summary.values()
    log_prob = np.array(log_prob)
    
    # Create a figure and axis
    fig, ax = plt.subplots()

    fig = plt.plot(log_prob[0], linewidth=0.05)
    plt.ylim(bottom=-20)
    mkdir(output_dir + "/likelihood_single_line")
    plt.savefig(output_dir + "/likelihood_single_line/"+event+".jpeg")

    # Plot each line
    for i in range(log_prob.shape[0]):
        ax.plot(log_prob[i], linewidth=0.05)
    plt.ylim(bottom=-20)
    mkdir(output_dir + "/likelihood")
    plt.savefig(output_dir + "/likelihood/"+event+".jpeg")
    
    



def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

############################## Scan through the GWTC-3 Database ##############################
def runAnalysis(event):
    file = Path(output_dir + '/posterior_samples/' + event + '.h5')
    if file.is_file():
        print(event + 'already exists')
    else:
        result, summary = runParameterEstimation(event)
        if result != None:
            mkdir(output_dir)
            plotPosterior(result, event)
            savePosterior(result, event)
            plotRunAnalysis(summary, event)
            plotLikelihood(summary, event)



if __name__ == '__main__':
    gwtc3 = datasets.find_datasets(type='events', catalog='GWTC-3-confident')
    event = "GW200129_065458-v1"
    runAnalysis(event)