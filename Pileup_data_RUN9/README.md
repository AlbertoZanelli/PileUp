#Pileup injection generation file
Those files are used to generate data used for pileup study. They are generate to follow the 0vbb spectrum and create set of single and pileup pulses.

- 'pup_n1-d0' files are the single pulses. Their columns are splited as 

binary position in the original binary file, amplitude, time smearing [1e-4 s]

- 'pup_n1-d8' files are the pileup pulses. Their columns are splited as

binary position in the original binary file, amplitude1, amplitude2, energy1, energy2, time smearing [1e-4 s], delta_t [1e-4 s]

The following '000813_20230628T161508_***_000' correspond to the name of the original file to be injected for the different channels.
