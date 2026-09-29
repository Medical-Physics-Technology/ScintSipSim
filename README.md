# sipmsalabim

Based on the original work of Stefan van der Sar, this codebase aims to simulate the generation of pulses from a scintillator coupled to a SiPM. The code takes in a number of occured events (or an event rate) at specified energies and simulates the output pulse(s) based on various statistical derivations. The simulation takes into account cross-talk as well as pile-up (both for SPADs within an event and for events within a trace). The code does *not* perform any ray or particle like tracing to determine deposited energy or optical cross-talk. Instead it accepts a deposited energy within the scintillator.

## The SiPM_Class

This is the heart of the simulation and contains the methods you call to run simulations with. It consists broadly of 2 sections: a list of defined scintillators and the SiPM class itself. The scintillator list is a static class containing information about scintilators to be coupled to a SiPM. Adding your own scintillator involves creating one within this section of the file. 

The input to the simulation is generally a scintillator (from the list of defined scintillators), a timestep, a number of events or and event rate, the length of the trace and an array of energies over which the events are spread.

The simulation starts by calculating the various probability density and cumulative density functions for the decays of the scintillator and the SPADs. These density functions are used to draw times for events later. The simulation loops over all events (incident energetic rays) and assigns to each SPAD a number of timestamps corresponding to when it is triggered. After all events have been handled, the SPADs are added up together to produce the final trace.

Within a single event the number of total scintillation triggers is drawn from a normal distribution with parameters from the SiPM and attached scintillator as well as the impinging energy. These triggers are uniformly dsitributed among the SPADs. Potential cross-talk events are then handled and any extra triggers are added to the relevant SPADs. Scintillation triggers are added to SPADs in such a way that they saturate, simulating pileup.

The output is properly scaled in terms of current. Be aware that the trace_time array is in terms of the timestep and needs to be scaled by timeScale to properly align. See example_trace.py for an example.

