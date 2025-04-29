# Spectrum TOP

This example implements a TOP to calculate the Fourier Transform of a TOP using OpenCV's cuda functionallity.

For more information on image Fourier Transforms [read this OpenCV article](https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html)



## Prerequisites
Requires a [reference](https://github.com/TouchDesigner/CustomOperatorSamples#referencing-opencv-libraries) to the OpenCV include and library folder.

Requires the CUDA Toolkit to be [installed](https://github.com/TouchDesigner/CustomOperatorSamples/blob/main/README.md#installing-the-cuda-toolkit).

## Parameters
* **Transform**:	One of [Image To DFT, DFT To Image], which determines if we calculate the forward or 
	inverse fourier transform.
* **Coordinate System**:	One of [Polar, Cartesian]. If the transform is Image To DFT the output will be
	in the selected coordinate system. If the transform is DFT To Image the input must be in the 
	coordinate system selected.
* **Channel**:	Active when Transform is Image To DFT. Selects which channel will be used to calculate the transform.
* **Per Rows**:	If On, it calculates the fourier transform of each row independently.

This TOP takes one input. If the inverse is to be calculated the input must have
exactly 2 32-bit float channels.
