import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time
import math
import pandas as pd
from decimal import Decimal
import random
import secrets

@dataclass()
class Scintillator:
    """ 
    Data class containing scintillator properties
    """
    name: str
    lightYield: int
    LCE: float
    amplitudes: list
    taus: list
    t0s: list
    RE_a: float
    RE_b: float
    RE_c: float
    nonProp: list
    E_arr_nonProp: list

# real life scintillators, add more as required
SCINTILLATORS = {
    "LaBr": Scintillator(
        name = "LaBr", lightYield = 63, LCE = 0.69, amplitudes = [1], taus = [16], t0s = [0],
        RE_a = 0, RE_b = 0.81, RE_c = -0.5,
        nonProp = [None],
        E_arr_nonProp = [None]
    ),
    "YAP": Scintillator(
        name = "YAP",  lightYield = 25, LCE = 0.56, amplitudes = [0.965, 0.035], taus = [26, 139], t0s = [0, 0],
        RE_a = 0.30, RE_b = 0, RE_c = -0.5,
        nonProp = [None],
        E_arr_nonProp = [None]
    ),
    "LYSO": Scintillator(
        name = "LYSO", lightYield = 33, LCE = 0.88, amplitudes = [1], taus = [36], t0s = [0],
        RE_a = 0, RE_b = 2.4, RE_c = -0.5,
        nonProp=[0.7308248, 0.791220556745182, 0.808779443254818, 0.8978586723768737,0.9068522483940042, 0.8533190578158458,0.9145610278372591,0.9775160599571735,
                 0.9847965738758031,0.99593147751606,1.00,1.0036402569593148,1.0044967880085653,1.000042398286938],
        E_arr_nonProp  = [14, 30.9, 32.0, 53.0, 59.9, 81.0, 160.6, 302.9, 356.0, 511.0, 661.7, 1173.2, 1274.5, 1332.5]
    ),
    "BZAPbBr": Scintillator( # VALUES NOT ACCURATE !!!!
        name = "BZAPbBr",  lightYield = 4, LCE = 0.56, amplitudes = [1], taus = [4], t0s = [0],
        RE_a = 0.30, RE_b = 0, RE_c = -0.5,
        nonProp = [None],
        E_arr_nonProp = [None]
    )
}

class Detector:
    """
    The Detector Class is intended to couple a Scintillator object to some arbitrary detector parameters in order to create a full detector.
    """

    def __init__(self, scintillator="LYSO", PDE=0.28, pixelSize=1000, sizeSPAD=15, prob_ct=0.08, spad_amplitudes=[1], spad_taus=[7], spad_t0s = [0], gain_SiPM=4e6):

        self.PDE             = PDE             # Photon detection efficiency, averaged over the emission spectrum of the scintillator
        self.pixelSize       = pixelSize       # In microns (assumes square)
        self.sizeSPAD        = sizeSPAD        # In microns (assumes square)
        self.nrSPADs_side    = int(self.pixelSize / self.sizeSPAD)        # SPADs on a side
        self.nrSPADs         = int(self.pixelSize / self.sizeSPAD)**2     # Number of SPADs on a pixel
        self.prob_ct         = prob_ct         # Crosstalk probability
        self.gain_SiPM       = gain_SiPM       # SiPM gain
        self.spad_amplitudes = spad_amplitudes # SPAD decay mode amplitudes (should sum to 1)
        self.spad_taus       = spad_taus       # Recharge time(s) in ns of the decay modes
        self.spad_t0s        = spad_t0s        # Decay mode offsets
        self.rangeE_RE       = np.arange(1,1001,1).astype(float)

        try:
            self.scintillator = SCINTILLATORS[scintillator]
        except KeyError:
            print(f"(!) The scintillator \"{scintillator}\" is not listed in the SCINTILLATORS as defined in SiPM_Class.py")
            print(f"(!) Assigning default scintillator LYSO.")
            self.scintillator = SCINTILLATORS["LYSO"]
        except:
            print("(!) There is something rotten in the state of Denmark (something else went wrong with scintillation assignment)")
            print(f"(!) Assigning default scintillator LYSO.")
            self.scintillator = SCINTILLATORS["LYSO"]

        if self.scintillator.nonProp[0] is None:
            self.nPfactor       = np.ones(1000)
            self.rangeE_nonProp = np.arange(1,1001,1)
        else:
            self.rangeE_nonProp = np.arange(1,self.scintillator.E_arr_nonProp[-1],1)
            self.nPfactor       = np.interp(self.rangeE_nonProp, self.scintillator.E_arr_nonProp, self.scintillator.nonProp)
        
        self.ran_prob     = False
        self.ran_sim      = False

    def R_E(self, a, b, c, E):
        """
        Energy efficiency. This is a measure of the FWHM of the pulse height distribution as a function of energy. It is used to determine the standard deviation of scintillation triggers for a given energy.
        """
        return a + b*E**c

    def Exp(self, x, a, tau, x0):
        return a*np.exp(-(x-x0)/tau)

    def fexp(self, number):
        """
        Returns the exponent of a number in scientific notation
        Source - https://stackoverflow.com/a/45359185
        Posted by jsalonen, modified by community. See post 'Timeline' for change history
        Retrieved 2026-08-27, License - CC BY-SA 3.0
        """
        (sign, digits, exponent) = Decimal(number).as_tuple()
        return len(digits) + int(exponent) - 1

    def fman(self, number):
        """
        Returns the mantissa of a number in scientific notation
        Source - https://stackoverflow.com/a/45359185
        Posted by jsalonen, modified by community. See post 'Timeline' for change history
        Retrieved 2026-08-27, License - CC BY-SA 3.0
        """
        return Decimal(number).scaleb(-self.fexp(number)).normalize()

    def prob_calc(self, timestep=1e-9, verbose=False):
        """ 
        Calculates the pdfs and cdfs of the decay, spad and convolved
        """

        self.timeStep     = timestep
        self.timeStepint  = int(self.fman(self.timeStep))
        self.timeStepexp  = self.fexp(self.timeStep)
        self.timeScale    = 1e-9/10**self.timeStepexp # [timestep/ns]

        # ns
        self.decayTime    = int(np.sum(self.scintillator.taus))
        self.rechargeTime = int(np.sum(self.spad_taus))

        # In timestep scale
        self.t            = np.arange(0, 10*(self.decayTime+self.rechargeTime)*self.timeScale+self.timeStepint, self.timeStepint).astype(int)

        if abs(np.sum(self.scintillator.amplitudes)-1) > 1e-6:
            print(f"(!) Warning: sum of decay amplitudes for {self.scintillator.name} is not equal to 1. It is {np.sum(self.scintillator.amplitudes)}.\n(+) Setting last decay amplitude to {1-np.sum(self.scintillator.amplitudes[:-1])}.")
            self.scintillator.amplitudes[-1] = 1-np.sum(self.scintillator.amplitudes[:-1])

        if abs(np.sum(self.spad_amplitudes)-1) > 1e-6:
            print(f"(!) Warning: sum of SPAD amplitudes for {self.scintillator.name} is not equal to 1. It is {np.sum(self.spad_amplitudes)}.\n(+) Setting last SPAD amplitude to {1-np.sum(self.spad_amplitudes[:-1])}.")
            self.spad_amplitudes[-1] = 1-np.sum(self.spad_amplitudes[:-1])

        # Generate the pdfs and cdfs of the decay, spad and convolved
        f_decay = np.zeros(len(self.t))
        for i in range(len(self.scintillator.amplitudes)):
            f_decay += self.Exp(self.t, self.scintillator.amplitudes[i], self.scintillator.taus[i]*self.timeScale, self.scintillator.t0s[i]*self.timeScale)

        f_spad = np.zeros(len(self.t))
        for i in range(len(self.spad_amplitudes)):
            f_spad += self.Exp(self.t, self.spad_amplitudes[i], self.spad_taus[i]*self.timeScale, self.spad_t0s[i]*self.timeScale)

        self.pdf_decay = f_decay/np.sum(f_decay) 
        self.cdf_decay = np.insert(np.cumsum(self.pdf_decay), 0, 0) # Leading 0 ensures that the CDF starts at 0 for 0 events
        self.cdf_decay_max = max(self.cdf_decay)

        self.pdf_spad = f_spad/np.sum(f_spad)
        self.cdf_spad = np.insert(np.cumsum(self.pdf_spad), 0, 0) # Leading 0 ensures that the CDF starts at 0 for 0 events
        self.cdf_spad_max = max(self.cdf_spad)

        f_conv = np.convolve(f_spad, f_decay, mode='full')
        f_conv = f_conv[0:len(self.t)]
        self.pdf_conv = f_conv / sum(f_conv)
        self.cdf_conv = np.insert(np.cumsum(self.pdf_conv), 0, 0) # Leading 0 ensures that the CDF starts at 0 for 0 events

        self.ran_prob = True

        if verbose:
            print("\n============ Checks ============")
            print(f'Sum of pdf_decay        = {np.sum(self.pdf_decay):.3f}')
            print(f'Last value in cdf_decay = {self.cdf_decay[-1]:.3f}')
            print(f'Sum of pdf_spad         = {np.sum(self.pdf_spad):.3f}')
            print(f'Last value in cdf_spad  = {self.cdf_spad[-1]:.3f}')
            print(f'Sum of pdf_conv         = {np.sum(self.pdf_conv):.3f}')
            print(f'Last value in cdf_conv  = {self.cdf_conv[-1]:.3f}')
            print("================================\n")

    def sim_single_pulses(self, N_events=1000, timestep=1e-9, verbose=False, E_arr = [100], seed=None):
        """
        Simulates N_events scintillation events with energies across defined range.
        Doesn't return anything explicitly but pulses are stored in the df_pulses attribute.
        """

        if not(self.ran_prob):
            print("(!) Probabilities not calculated, running prob_calc()")
            self.prob_calc(timestep=timestep, verbose=verbose)

        # Set up RNG
        if seed == None: seed = secrets.randbits(128)
        rng = np.random.default_rng(seed)

        # Parameters
        self.N_events = N_events
        N_steps       = len(self.t)

        # Neighbour set up for crosstalk stuff
        n_neigbors      = 8
        left_row_SPADs  = np.arange(0, self.nrSPADs, self.nrSPADs_side)
        right_row_SPADs = np.arange(self.nrSPADs_side-1, self.nrSPADs, self.nrSPADs_side)
        neighbors       = np.array([[SPADi-self.nrSPADs_side-1, SPADi-self.nrSPADs_side, SPADi-self.nrSPADs_side+1,  # neighbors[i] is the list of neighbors for SPAD i,
                                    SPADi-1             ,                     SPADi+1,                               # in order of from top left to bottom right, skipping the SPAD itself
                                    SPADi+self.nrSPADs_side-1, SPADi+self.nrSPADs_side, SPADi+self.nrSPADs_side+1] for SPADi in range(self.nrSPADs)])

        # Borel distribution for crosstalk. See Vinogradov 2011 Analytical models of probability distribution and excess noise factor of solid state photomultiplier signals with crosstalk
        lambdaa  = self.prob_ct/(1+self.prob_ct)
        n_ct_max = 6
        ns       = np.arange(1, n_ct_max+1)

        pdf_borrel      = np.exp(-lambdaa*ns)*(lambdaa*ns)**(ns-1)/np.array([math.factorial(n_i) for n_i in ns]) # From Table 2 Vinogradov 2011
        cdf_borrel      = np.insert(np.cumsum(pdf_borrel), 0, 0) # CDF of the Borel distribution, leading 0 ensures that the CDF starts at 0 for 0 events
        cdf_borrel_max  = np.max(cdf_borrel)

        # Variable setup
        outputPulses    = np.zeros((N_events,N_steps))             
        self.xRayEnergy = np.zeros(N_events, dtype=int)
        n_tr_sc         = np.zeros(N_events, dtype=int)
        R_Ev            = self.R_E(self.scintillator.RE_a, self.scintillator.RE_b, self.scintillator.RE_c, self.rangeE_RE) # energy resolution at given energies

        # Isolated SPAD pulse generation
        # Based on Schaart 2021 Physics and Technology of Time-Of-Flight PET Detectors Eq. 24
        Q        = self.gain_SiPM * 1.6e-19
        SSR      = Q/(self.rechargeTime*1e-9)*np.exp(-self.t/(self.rechargeTime*self.timeScale))
        self.SSR = SSR # Single Spad Response

        # Info
        if verbose:
            print(f"\n(+) Scintillator     : {self.scintillator.name}")
            print(f"(+) Number of events : {N_events}")
            print(f"(+) Energies [keV]   : {E_arr}")
            print(f"(+) Time step [s]    : {self.timeStep:.3e}")
            print(f"(+) Random Seed      : {seed}")

        event_out = int(np.ceil(0.01*N_events)) # flag for printing progress every 1% of events

        print("\n=========== Starting Simulation ===========")

        t0 = time.time()

        # Loop over all scintillation events
        for i in range(N_events):
            # Print Progress
            if (i % event_out == 0) and verbose:
                print(f"(+) Event {i} of {N_events} ({int(i/N_events*100)}%)",end="\r")
            
            # Equally distribute the energies of the events over the energy range
            self.xRayEnergy[i] = E_arr[i % len(E_arr)]
            
            # Scintillation triggers based on normal distribution (van der Sar et al: SiPM-based detectors for photon-counting CT)
            mu            = self.xRayEnergy[i] * self.scintillator.lightYield * self.nPfactor[self.xRayEnergy[i]-1] * self.scintillator.LCE * self.PDE
            std           = R_Ev[self.xRayEnergy[i]-1]*mu/2.355
            n_tr_sc[i]    = rng.normal(mu, std)

            if verbose and mu<10:
                print(f"(!) mean number of scintillation triggers {mu:.3f} is less than 10, approximating Poisson as Guass is questionable...")
            
            # If negative trigger (due to low mu or energy resolution) just say 1 occured and note it
            if n_tr_sc[i]<1:
                print(f'(!) Number of scintillation triggers < 1 for event {i}. Set to 1.')
                print(f"(!) Energy = {self.xRayEnergy[i]:.3f}, Energy Resolution: {R_Ev[self.xRayEnergy[i]-1]:.3f}, mu: {mu:.3f}, std: {std:.3f}")
                n_tr_sc[i]  = 1

            # Setup pulse and number of fired SPADs
            outputPulse = np.zeros(N_steps)

            # Integer number of triggers total
            n      = int(n_tr_sc[i])
            self.n = n

            # Generate trigger times and SPAD assignments for n triggers
            p_random = rng.uniform(0, self.cdf_decay_max, size=n)
            ts_tr_sc = np.searchsorted(self.cdf_decay, p_random, side='right') * self.timeStepint # Looks for where p_random>cdf_decay, these are the timestamps of the n triggers
            idx      = rng.integers(0, self.nrSPADs, size=n)                   # idx[i] = SPAD id for trigger i
            arr      = np.bincount(idx, minlength=self.nrSPADs)                # arr[i] = number of triggers assigned to SPAD i

            # Sort timestamps by SPAD index so they're grouped correctly.
            sort_order      = np.argsort(idx, kind='stable') # sort_order[i] = index of the i-th trigger in the sorted order by SPAD index
            ts_tr_sc_sorted = ts_tr_sc[sort_order]  # ts_tr_sc_sorted[i] = timestamp of the i-th trigger in the sorted order by SPAD index

            # Split sorted timestamps into per-SPAD lists
            split_points = np.cumsum(arr)[:-1] # since arr contains the number of triggers per SPAD, the cumsum shows the indices where the sorted timestamps should be split
            ts_list      = np.split(ts_tr_sc_sorted, split_points) # So we split them
            ts_list_all  = [ts.tolist() for ts in ts_list]         # Convert it to a list of lists, each sublist contains the timestamps of the triggers for that SPAD, in order of SPAD index

            # Map each trigger to its SPAD and timestamp
            trigger_spad = np.repeat(np.arange(self.nrSPADs), arr) # SPAD id for each of the n triggers. Equivalent to sorting the idx array by SPAD index, but more efficient since we already have the counts in arr
            trigger_ts   = ts_tr_sc_sorted                         # timestamp for each of the n triggers

            # Number of crosstalk events based on Borel distribution, using same vectorization trick as before with searchsorted
            u    = rng.uniform(0, cdf_borrel_max, size=n)
            k    = np.searchsorted(cdf_borrel, u, side='right') # Looks for where u>cdf_borrel, these are the number of crosstalk events for each trigger
            n_ct = k - 1  # -1 since 1 crosstalk event is actually just the initial trigger, so we subtract 1 to get the number of additional crosstalk events

            # Expand to one row per individual crosstalk event
            total_ct    = n_ct.sum()
            parent_idx  = np.repeat(np.arange(n), n_ct) # parent_idx[i]  = index of the parent trigger for crosstalk event i
            parent_spad = trigger_spad[parent_idx]      # parent_spad[i] = SPAD id of the parent trigger for crosstalk event i
            parent_ts   = trigger_ts[parent_idx]        # parent_ts[i]   = timestamp of the parent trigger for crosstalk event i

            neighbor = rng.integers(0, n_neigbors, endpoint=False, size=total_ct) # Pick a random (relative) neighbor to crosstalk with
            
            #################
            #   0   1   2   #
            #   3  SPAD 4   #
            #   5   6   7   #
            #################

            # Vectorized validity check. We get a valid and invalid mask which we can use to find chosen SPADs.
            # | is an or operator here.
            invalid = (
                ((neighbor <= 2) & (parent_spad < self.nrSPADs_side))                    # top row
                | ((neighbor >= 5) & (parent_spad >= self.nrSPADs - self.nrSPADs_side))  # bottom row
                | (np.isin(neighbor, [0, 3, 5]) & np.isin(parent_spad, left_row_SPADs))  # left column
                | (np.isin(neighbor, [2, 4, 7]) & np.isin(parent_spad, right_row_SPADs)) # right column
            )
            valid = ~invalid # Valid is (bitwise) not invalid. Cogito ergo sum

            # Recall that neighbors contains the SPAD ids of the neighbors for each SPAD
            # in order from top left to bottom right, skipping the SPAD itself.
            # So neighbors[parent_spad[i], neighbor[i]] gives the SPAD id of the neighbor that crosstalk event i is assigned to.
            idx        = neighbors[parent_spad[valid], neighbor[valid]] # idx[i] = SPAD id of the neighbor that crosstalk event i is assigned to
            ts_to_add  = parent_ts[valid]                               # Add parent timestamp since it is an 'instant' crosstalk event

            # Append crosstalk timestamps to the right SPAD's list, grouped in one pass
            # order within each SPAD's list doesn't matter since it's np.sort()ed before use later
            # if idx.size > 0 ensures we only run if there are actually cross talk events to add
            if idx.size > 0:
                sort_order           = np.argsort(idx, kind='stable')
                ts_sorted            = ts_to_add[sort_order]  # Sort the crosstalk timestamps by crosstalk target SPAD index
                unique_spads, counts = np.unique(idx, return_counts=True) # unique_spads = the SPADs that received crosstalk events
                                                                          # counts       = how many crosstalk events each of those SPADs received
                split_points         = np.cumsum(counts)[:-1]             # Where to split the sorted timestamps to get per-SPAD lists
                chunks               = np.split(ts_sorted, split_points)  # Split the sorted timestamps into per-SPAD lists

                # loop over the crosstalk target SPADs and append the crosstalk timestamps (chunks) to the right SPAD's list
                for spad_id, chunk in zip(unique_spads, chunks):
                    spad_ts              = np.asarray(ts_list_all[spad_id]) # Get the existing timestamps for that SPAD
                    ts_list_all[spad_id] = np.concatenate([spad_ts, chunk]) # Append the crosstalk timestamps to the existing list for that SPAD

            # Generate the time axis for the output
            t_axis = self.t
            
            # Only loop over SPADs that actually fired
            active_spads = [s for s in range(self.nrSPADs) if len(ts_list_all[s]) > 0]

            for SPADi in active_spads:

                ts     = np.sort(ts_list_all[SPADi]).astype(int) # Sort the timestamps for this SPAD

                # owner[i] = index of the last timestamp in ts that is less than or equal to t_axis[i]
                # Example:
                # ts     = [        2,       5,       8   ]
                # t_axis = [ 0,  1, 2, 3, 4, 5, 6, 7, 8, 9]
                # owner  = [-1, -1, 0, 0, 0, 1, 1, 1, 2, 2]
                owner  = np.searchsorted(ts, t_axis, side='right') - 1
                mask   = owner >= 0 # Mask away the beginning of the trace where no triggers have occured

                # SSR_offset[i] = How far into the Single Spad Response we are for the trigger that owns t_axis[i].
                # This is the index into Single Spad Response that we will add into outputPulse at t_axis[i].
                # Doing it this way ensures we have proper saturation of the SPADs, since they don't just
                # naively add up. Note that this only works because the SPAD response starts at its
                # max value and then decays monotonically. If there was a rise time we would see very non physical behavior here.
                SSR_offset = (t_axis[mask] - ts[owner[mask]]).astype(int) 

                # In the case that the timestep mantissa is not 1,
                # we need to scale the SSR_offset by the timestep mantissa to get the correct index into the SSR array
                # this because the SSR is generated assuming timestep mantissa of 1
                SSR_offset = SSR_offset // self.timeStepint # // is a floor division by the way. A ceil would also work.

                outputPulse[mask] += SSR[SSR_offset] # Add the SPAD response to the output pulse at the right times, with the right offsets

            outputPulses[i, :] = outputPulse    # Store the output pulse for this event

        # Generate a pandas dataframe with the output pulses, with a leading column of zeros for the time axis
        self.outputPulses2 = np.insert(outputPulses, 0, 0, axis=1)
        self.df_pulses = pd.DataFrame(self.outputPulses2)
        t1 = time.time()

        if verbose:
            print(f"(+) Event {i+1} of {N_events} (100%)\n")
        print(f"(+) Total time {t1-t0:.3f} seconds at {(t1-t0)/N_events*1000:.3f} ms per event.")
        print("=========== End of Simulation ===========\n")

        self.ran_sim = True

    def sim_pulse_trace(self, N_events=None, timestep=1e-9, event_rate=10e6, trace_length=10e-6, verbose=False, E_arr = [100], seed=None):
        """
        Simulates a pulse trace of length trace_length.
        You can specify an event rate or give a specific number of events (N_events)
        """

        self.prob_calc(timestep=timestep, verbose=verbose)

        # Set up RNG
        if seed == None: seed = secrets.randbits(128)
        rng = np.random.default_rng(seed)

        # Parameters
        self.event_rate   = event_rate   # Events per second
        self.trace_length = trace_length # In seconds
        self.trace_time   = np.arange(0,self.trace_length/10**self.timeStepexp,self.timeStepint).astype(int) # In unit of timestep
        N_steps           = len(self.trace_time)

        # Neighbour set up for crosstalk stuff
        n_neigbors      = 8
        left_row_SPADs  = np.arange(0, self.nrSPADs, self.nrSPADs_side)
        right_row_SPADs = np.arange(self.nrSPADs_side-1, self.nrSPADs, self.nrSPADs_side)
        neighbors       = np.array([[SPADi-self.nrSPADs_side-1, SPADi-self.nrSPADs_side, SPADi-self.nrSPADs_side+1,  # neighbors[i] is the list of neighbors for SPAD i,
                                    SPADi-1             ,                     SPADi+1,                               # in order of from top left to bottom right, skipping the SPAD itself
                                    SPADi+self.nrSPADs_side-1, SPADi+self.nrSPADs_side, SPADi+self.nrSPADs_side+1] for SPADi in range(self.nrSPADs)])

        # Borel distribution for crosstalk. See Vinogradov 2011 Analytical models of probability distribution and excess noise factor of solid state photomultiplier signals with crosstalk
        lambdaa  = self.prob_ct/(1+self.prob_ct)
        n_ct_max = 6
        ns       = np.arange(1, n_ct_max+1)

        pdf_borrel      = np.exp(-lambdaa*ns)*(lambdaa*ns)**(ns-1)/np.array([math.factorial(n_i) for n_i in ns]) # From Table 2 Vinogradov 2011
        cdf_borrel      = np.insert(np.cumsum(pdf_borrel), 0, 0) # CDF of the Borel distribution, leading 0 ensures that the CDF starts at 0 for 0 events
        cdf_borrel_max  = np.max(cdf_borrel)

        # Events
        
        # If no N_events is specified, we draw from a Poisson distribution with mean = event_rate * trace_length
        if N_events == None:
            N_events = rng.poisson(self.event_rate*self.trace_length)
        if N_events == 0: N_events = 1 # Ensure at least one event occurs

        event_times = rng.integers(0, self.trace_time[-1], size=N_events, endpoint=True) # Random event times in order of timestep
        event_times = np.ceil(event_times/self.timeStepint)*self.timeStepint             # ceil to nearest timestep
        self.event_times = event_times

        # Variable setup
        pulse_trace     = np.zeros(N_steps) # The final pulse trace that will be returned
        self.xRayEnergy = np.zeros(N_events, dtype=int)
        n_tr_sc         = np.zeros(N_events, dtype=int)
        R_Ev            = self.R_E(self.scintillator.RE_a, self.scintillator.RE_b, self.scintillator.RE_c, self.rangeE_RE) # energy resolution at given energies

        # Isolated SPAD pulse generation
        # Based on Schaart 2021 Physics and Technology of Time-Of-Flight PET Detectors Eq. 24 (single exponential)
        Q        = self.gain_SiPM * 1.6e-19
        SSR      = Q/(self.rechargeTime*1e-9)*np.exp(-self.trace_time/(self.rechargeTime*self.timeScale))
        self.SSR = SSR

        # Info
        if verbose:
            print(f"=========== Info ===========")
            print(f"(+) Scintillator     : {self.scintillator.name}")
            print(f"(+) Number of events : {N_events}")
            print(f"(+) Energies [keV]   : {E_arr}")
            print(f"(+) Time step [s]    : {self.timeStep:.3e}")
            print(f"(+) Random Seed      : {seed}")
            print(f"============================")

        event_out = int(np.ceil(0.01*N_events)) # flag for printing progress every 1% of events

        print("\n=========== Starting Simulation ===========")

        t0 = time.time()

        ts_list_all = [[] for _ in range(self.nrSPADs)]

        # Loop over all scintillation events
        for i in range(N_events):
            # Print Progress
            if (i % event_out == 0) and verbose:
                print(f"(+) Event {i+1} of {N_events} ({int((i+1)/N_events*100)}%)",end="\r")
            
            # Equally distribute the energies of the events over the energy range
            self.xRayEnergy[i] = E_arr[i % len(E_arr)]

            # Time offset of the event (when it occured in the pulse trace)
            event_time = event_times[i]
            
            # Scintillation triggers based on normal distribution (van der Sar et al: SiPM-based detectors for photon-counting CT)
            mu            = self.xRayEnergy[i] * self.scintillator.lightYield * self.nPfactor[self.xRayEnergy[i]-1] * self.scintillator.LCE * self.PDE
            std           = R_Ev[self.xRayEnergy[i]-1]*mu/2.355
            n_tr_sc[i]    = rng.normal(mu, std)

            if verbose and mu<10:
                print(f"(!) mean number of scintillation triggers {mu:.3f} is less than 10, approximating Poisson as Guass is questionable...")
            
            # If negative trigger (due to low mu or relative efficiency) just say 1 occured and note it
            if n_tr_sc[i]<1:
                print(f'(!) Number of scintillation triggers < 1 for event {i}. Set to 1.')
                print(f"(!) Energy = {self.xRayEnergy[i]:.3f}, Relative Efficiency: {R_Ev[self.xRayEnergy[i]-1]:.3f}, mu: {mu:.3f}, std: {std:.3f}")
                n_tr_sc[i]  = 1

            # Integer number of triggers total
            n = int(n_tr_sc[i])
            self.n = n

            # Generate trigger times and SPAD assignments for n triggers
            p_random = rng.uniform(0, self.cdf_decay_max, size=n)
            ts_tr_sc = np.searchsorted(self.cdf_decay, p_random, side='right')* self.timeStepint + event_time # Looks for where p_random>cdf_decay, these are the timestamps of the n triggers
            idx      = rng.integers(0, self.nrSPADs, size=n)    # idx[i] = SPAD id for trigger i
            arr      = np.bincount(idx, minlength=self.nrSPADs) # arr[i] = number of triggers assigned to SPAD i

            # Sort timestamps by SPAD index so they're grouped correctly.
            sort_order      = np.argsort(idx, kind='stable') # sort_order[i] = index of the i-th trigger in the sorted order by SPAD index
            ts_tr_sc_sorted = ts_tr_sc[sort_order]  # ts_tr_sc_sorted[i] = timestamp of the i-th trigger in the sorted order by SPAD index

            # Split sorted timestamps into per-SPAD lists
            split_points = np.cumsum(arr)[:-1] # since arr contains the number of triggers per SPAD, the cumsum shows the indices where the sorted timestamps should be split
            ts_list      = np.split(ts_tr_sc_sorted, split_points) # So we split them

            # Now we append the time stamps to our global list
            # this step differs from the single pulse generation since we 
            # need to keep track of all the timestamps for each SPAD across all events, not just the current event
            # this to ensure we get proper saturation of SPADs in case events overlap
            for spad_id, chunk in enumerate(ts_list):
                if len(chunk) > 0:
                    spad_ts              = np.asarray(ts_list_all[spad_id]) # Get the existing timestamps for that SPAD
                    ts_list_all[spad_id] = np.concatenate([spad_ts, chunk]) # Append the crosstalk timestamps to the existing list for that SPAD

            ##### Crosstalk Generation #####

            # Map each trigger to its SPAD and timestamp
            trigger_spad = np.repeat(np.arange(self.nrSPADs), arr) # SPAD id for each of the n triggers. Equivalent to sorting the idx array by SPAD index, but more efficient since we already have the counts in arr
            trigger_ts   = ts_tr_sc_sorted                         # timestamp for each of the n triggers

            # Number of crosstalk events based on Borel distribution, using same vectorization trick as before with searchsorted
            u    = rng.uniform(0, cdf_borrel_max, size=n)
            k    = np.searchsorted(cdf_borrel, u, side='right') # Looks for where u>cdf_borrel, these are the number of crosstalk events for each trigger
            n_ct = k - 1  # -1 since 1 crosstalk event is actually just the initial trigger, so we subtract 1 to get the number of additional crosstalk events

            # Expand to one row per individual crosstalk event
            total_ct    = n_ct.sum()
            parent_idx  = np.repeat(np.arange(n), n_ct) # parent_idx[i]  = index of the parent trigger for crosstalk event i
            parent_spad = trigger_spad[parent_idx]      # parent_spad[i] = SPAD id of the parent trigger for crosstalk event i
            parent_ts   = trigger_ts[parent_idx]        # parent_ts[i]   = timestamp of the parent trigger for crosstalk event i

            neighbor = rng.integers(0, n_neigbors, size=total_ct) # Pick a random (relative) neighbor to crosstalk with
            
            #################
            #   0   1   2   #
            #   3  SPAD 4   #
            #   5   6   7   #
            #################

            # Vectorized validity check. We get a valid and invalid mask which we can use to find chosen SPADs.
            # | is an 'or' operator here.
            invalid = (
                ((neighbor <= 2) & (parent_spad < self.nrSPADs_side))                    # top row
                | ((neighbor >= 5) & (parent_spad >= self.nrSPADs - self.nrSPADs_side))  # bottom row
                | (np.isin(neighbor, [0, 3, 5]) & np.isin(parent_spad, left_row_SPADs))  # left column
                | (np.isin(neighbor, [2, 4, 7]) & np.isin(parent_spad, right_row_SPADs)) # right column
            )
            valid = ~invalid # Valid is (bitwise) not invalid. Cogito ergo sum

            # Recall that neighbors contains the SPAD ids of the neighbors for each SPAD
            # in order from top left to bottom right, skipping the SPAD itself.
            # So neighbors[parent_spad[i], neighbor[i]] gives the SPAD id of the neighbor that crosstalk event i is assigned to.
            idx        = neighbors[parent_spad[valid], neighbor[valid]] # idx[i] = SPAD id of the neighbor that crosstalk event i is assigned to
            ts_to_add  = parent_ts[valid]                               # Add parent timestamp since it is an 'instant' crosstalk event

            # Append crosstalk timestamps to the right SPAD's list, grouped in one pass
            # order within each SPAD's list doesn't matter since it's np.sort()ed before use later
            # if idx.size > 0 ensures we only run if there are actually cross talk events to add
            if idx.size > 0:
                sort_order           = np.argsort(idx, kind='stable')
                ts_sorted            = ts_to_add[sort_order]  # Sort the crosstalk timestamps by crosstalk target SPAD index
                unique_spads, counts = np.unique(idx, return_counts=True) # unique_spads = the SPADs that received crosstalk events
                                                                          # counts       = how many crosstalk events each of those SPADs received
                split_points         = np.cumsum(counts)[:-1]             # Where to split the sorted timestamps to get per-SPAD lists
                chunks               = np.split(ts_sorted, split_points)  # Split the sorted timestamps into per-SPAD lists

                # loop over the crosstalk target SPADs and append the crosstalk timestamps (chunks) to the right SPAD's list
                for spad_id, chunk in zip(unique_spads, chunks):
                    spad_ts              = np.asarray(ts_list_all[spad_id]) # Get the existing timestamps for that SPAD
                    ts_list_all[spad_id] = np.concatenate([spad_ts, chunk]) # Append the crosstalk timestamps to the existing list for that SPAD

        if verbose:
            print(f"(+) Event {N_events} of {N_events} (100%)")
        
        # Only loop over SPADs that actually fired
        active_spads = [s for s in range(self.nrSPADs) if len(ts_list_all[s]) > 0]

        # Keep track of total progress
        prog      = 0
        n_spads   = len(active_spads)
        event_out = int(np.ceil(0.01*n_spads)) # flag for printing progress every 1% of events
        

        for SPADi in active_spads:

            if (prog % event_out == 0) and verbose:
                print(f"(+) SPAD {prog+1} of {n_spads} ({int((prog+1)/n_spads*100)}%)",end="\r")

            ts     = np.sort(ts_list_all[SPADi]).astype(int) # Sort the timestamps for this SPAD

            # owner[i] = index of the last timestamp in ts that is less than or equal to t_axis[i]
            # Example:
            # ts     = [        2,       5,       8   ]
            # t_axis = [ 0,  1, 2, 3, 4, 5, 6, 7, 8, 9]
            # owner  = [-1, -1, 0, 0, 0, 1, 1, 1, 2, 2]
            owner  = np.searchsorted(ts, self.trace_time, side='right') - 1
            mask   = owner >= 0 # Mask away the beginning of the trace where no triggers have occured

            # SSR_offset[i] = How far into the Single Spad Response we are for the trigger that owns t_axis[i].
            # This is the index into Single Spad Response that we will add into outputPulse at t_axis[i].
            # Doing it this way ensures we have proper saturation of the SPADs, since they don't just
            # naively add up. Note that this only works because the SPAD response starts at its
            # max value and then decays monotonically. If there was a rise time we would see very non physical behavior here.
            SSR_offset = (self.trace_time[mask] - ts[owner[mask]]).astype(int)

            # In the case that the timestep mantissa is not 1,
            # we need to scale the SSR_offset by the timestep mantissa to get the correct index into the SSR array
            SSR_offset = SSR_offset // self.timeStepint # // is a floor division by the way


            pulse_trace[mask] += SSR[SSR_offset] # Add the SPAD response to the pulse trace at the right times, with the right offsets
            prog +=1 # Add to progress ticker

        self.df_pulses = pd.DataFrame(pulse_trace)
        t1 = time.time()

        if verbose:
            print(f"(+) SPAD {n_spads} of {n_spads} (100%)")
        print(f"\n(+) Total time {t1-t0:.3f} seconds at {(t1-t0)/N_events*1000:.3f} ms per event.")
        print("=========== End of Simulation ===========\n")

        self.ran_sim = True

        # Variable Access
        self.seed    = seed

    def plot_probs(self):
        """
        Plots the pdfs and cdfs of the decay, spad and convolved
        """

        if self.ran_prob == False:
            self.prob_calc()
        
        fig, axs = plt.subplots(1, 2, sharey=False, figsize = (15,5))
        for i in range(2):
            axs[i].plot(self.t/self.timeScale,self.pdf_decay, color='blue', label = 'pdf_decay')
            axs[i].plot(self.t/self.timeScale,self.pdf_spad, color='orange', label = 'pdf_spad')
            axs[i].plot(self.t/self.timeScale,self.pdf_conv, color='green', label = 'pdf_conv')
            axs[i].spines['left'].set_color('blue')
            axs[i].yaxis.label.set_color('blue')
            axs[i].tick_params(axis='y', colors='blue')
            axs[i].set_xlabel('t [ns]')
            axs[i].set_ylabel('Probability')
            axs[i].legend(loc='center right') 
        for ax in axs:
            # create a twin of the axis that shares the x-axis
            ax2 = ax.twinx()
            # plot some data on each axis.
            ax2.plot(self.t/self.timeScale,self.cdf_decay, color='red')
            ax2.set_ylabel('Cumulative probability')
            ax2.spines['right'].set_color('red')
            ax2.yaxis.label.set_color('red')
            ax2.tick_params(axis='y', colors='red')
        axs[1].set_xlim(-5, 100)
        axs[1].set_ylim(0, 1.1*max(self.pdf_conv))
        fig.tight_layout()
        plt.show()
