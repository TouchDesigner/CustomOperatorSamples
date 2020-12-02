# One Euro CHOP

The One Euro CHOP implements the One Euro Filter as described here: http://cristal.univ-lille.fr/~casiez/1euro/
After initially making this custom operator, the functionality was also included into the build in [Filter CHOP](https://docs.derivative.ca/Filter_CHOP)
The one Euro Filter is especially useful when a person is in an interaction loop with TouchDeigner and wants quick response: It responds quickly to large changes in value, and it smooths out jitters in the input.

## Parameters
* **Cutoff Frequency (Hz)** - Decrease it if slow speed jitter is a problem.
* **Speed Coefficient** - Avoids high derivative bursts caused by jitter. (The research paper implementation fixes this value to 1Hz but defaults to )
* **Slope Cutoff Frequency (Hz)** - Increase if high speed lag is a problem.
