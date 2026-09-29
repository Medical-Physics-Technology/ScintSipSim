import numpy as np
import matplotlib.pyplot as plt
import SiPM_Class as sipm

# Create the detector object with a chosen scintillator
# new scintillators can be added to the class in the source file
det = sipm.Detector("LaBr",sizeSPAD=15)

# Run a pulse trace
det.sim_pulse_trace(timestep=4e-10, event_rate=1e7, trace_length=1e-6, verbose=True, E_arr=[50,110])

# Extract the trace
pulse_trace = det.df_pulses[0]

# The time steps will depend on the chosen timestep
# the code below converts it to the proper nanosecond scale
time_ns = det.trace_time/det.timeScale


# Plot the trace
plt.plot(time_ns,pulse_trace*1e3)
plt.xlabel("Time [ns]")
plt.ylabel("Current [mA]")
plt.show()