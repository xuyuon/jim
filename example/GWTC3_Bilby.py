import bilby
from gwpy.timeseries import TimeSeries
import time
from gwosc.datasets import find_datasets, event_gps
from gwosc import datasets
from jimgw.single_event.detector import H1, L1, V1
from tap import Tap

class ArgumentParser(Tap):
    label: str = "GW191103"
    index: int = 0

gwtc3 = datasets.find_datasets(type='events', catalog='GWTC-3-confident')

args = ArgumentParser().parse_args()

logger = bilby.core.utils.logger
outdir = "Bilby"
label = gwtc3[args.index]

def isPresentInH1(event):
    if event in datasets.find_datasets(type='events', catalog='GWTC-3-confident', detector='H1'):
        return True
    return False

def isPresentInL1(event):
    if event in datasets.find_datasets(type='events', catalog='GWTC-3-confident', detector='L1'):
        return True
    return False

def isPresentInV1(event):
    if event in datasets.find_datasets(type='events', catalog='GWTC-3-confident', detector='V1'):
        return True
    return False

def runParameterEstimation(event):
    ############################## Fetch event data ##############################
    total_time_start = time.time()

    # first, fetch a 4s segment centered on the event
    trigger_time = event_gps(event) # trigger time
    duration = 128 # Analysis segment duration
    post_trigger_duration = 2
    end_time = trigger_time + post_trigger_duration
    start_time = end_time - duration
    minimum_frequency = 20.0
    maximum_frequency = 1024.0
    

    psd_alpha = 0.05  # Roll off duration of tukey window in seconds, default is 0.4s

    psd_pad = 4*duration

    psd_duration = 32 * duration
    psd_start_time = trigger_time - duration + post_trigger_duration - 2*psd_pad
    # psd_start_time = start_time - psd_duration
    psd_end_time = trigger_time - duration +post_trigger_duration -psd_pad
    # psd_end_time = start_time
    
    
    detectors = []
    if isPresentInH1(event):
        detectors.append("H1")
    
    if isPresentInL1(event):
        detectors.append("L1")
        
    if isPresentInV1(event):
        detectors.append("V1")
    
    if len(detectors) == 0:
        print("No detector data from" + event)
    
    # We now use gwpy to obtain analysis and psd data and create the ifo_list
    ifo_list = bilby.gw.detector.InterferometerList([])
    for det in detectors:
        logger.info("Downloading analysis data for ifo {}".format(det))
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        if det == "H1":
            data, psd_data = H1.load_data(trigger_time, duration-post_trigger_duration, post_trigger_duration, minimum_frequency, maximum_frequency, psd_pad=4*duration, tukey_alpha=0.05)
        if det == "L1":
            data, psd_data = L1.load_data(trigger_time, duration-post_trigger_duration, post_trigger_duration, minimum_frequency, maximum_frequency, psd_pad=4*duration, tukey_alpha=0.05)
        if det == "V1":
            data, psd_data = V1.load_data(trigger_time, duration-post_trigger_duration, post_trigger_duration, minimum_frequency, maximum_frequency, psd_pad=4*duration, tukey_alpha=0.05)
        
        ifo.strain_data.set_from_gwpy_timeseries(data)

        logger.info("Downloading psd data for ifo {}".format(det))
        psd = psd_data.psd(
            fftlength=duration, overlap=0, window=("tukey", psd_alpha), method="median"
        )
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=psd.frequencies.value, psd_array=psd.value
        )
        ifo.maximum_frequency = maximum_frequency
        ifo.minimum_frequency = minimum_frequency
        ifo_list.append(ifo)
        
        
    logger.info("Saving data plots to {}".format(outdir))
    bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
    ifo_list.plot_data(outdir=outdir, label=label)

    # We now define the prior.
    # We have defined our prior distribution in a local file, GW150914.prior
    # The prior is printed to the terminal at run-time.
    # You can overwrite this using the syntax below in the file,
    # or choose a fixed value by just providing a float value as the prior.
    priors = bilby.gw.prior.BBHPriorDict(filename="default_prior.prior")

    # Add the geocent time prior
    priors["geocent_time"] = bilby.core.prior.Uniform(
        trigger_time - 0.05, trigger_time + 0.05, name="geocent_time"
    )

    # In this step we define a `waveform_generator`. This is the object which
    # creates the frequency-domain strain. In this instance, we are using the
    # `lal_binary_black_hole model` source model. We also pass other parameters:
    # the waveform approximant and reference frequency and a parameter conversion
    # which allows us to sample in chirp mass and ratio rather than component mass
    waveform_generator = bilby.gw.WaveformGenerator(
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={
            "waveform_approximant": "IMRPhenomPv2",
            "reference_frequency": 50,
        },
    )

    # In this step, we define the likelihood. Here we use the standard likelihood
    # function, passing it the data and the waveform generator.
    # Note, phase_marginalization is formally invalid with a precessing waveform such as IMRPhenomPv2
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list,
        waveform_generator,
        priors=priors,
        time_marginalization=True,
        phase_marginalization=False,
        distance_marginalization=True,
    )

    # Finally, we run the sampler. This function takes the likelihood and prior
    # along with some options for how to do the sampling and how to save the data
    result = bilby.run_sampler(
    likelihood,
    priors,
    sampler="dynesty",
    outdir=outdir,
    label=label,
    nlive=1000,
    check_point_delta_t=600,
    check_point_plot=True,
    npool=4,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
    )  
    result.plot_waveform_posterior(n_samples=1000)
    result.plot_corner()
    
    return result



runParameterEstimation(gwtc3[args.index])

