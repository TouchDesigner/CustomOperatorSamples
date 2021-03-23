# Timeslice Generator CHOP
This is a bare bones example for creating a timesliced generator CHOP. It generates one channel with one sample that is changing over time based on the type of waveform selected with the Type parameter. It is a simpler version of TouchDesigner built-in LFO CHOP.

## Parameters
* **Type** - The shape of the waveform to repeat
  * **Sine** - (-1 to 1) A Sine wave.
  * **Square** - (-1 to 1) Step-up/step-down.
  * **Ramp** - (0 to 1) A ramp from 0 to 1.
* **Frequency** - Frequency of the selected curve type.
* **Apply Scale** - Toggle on to have acces to the scale multiplier value below and apply scale to the current value.
* **Scale** - When Apply Scale is toggled on, scale the data by the specified value.
