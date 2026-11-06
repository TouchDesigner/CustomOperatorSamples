# Optical Flow (CPU) TOP

This example implements a TOP to expose cv::cuda::FarnebackOpticalFlow class functionallity. 

For more information on the parameters check 
https://docs.opencv.org/3.4/d9/d30/classcv_1_1cuda_1_1FarnebackOpticalFlow.html

## Prerequisites
Uses the OpenCV that ships with TouchDesigner. CMake finds it in the TouchDesigner installation set by `TD_INSTALLATION_PATH`, see [Referencing OpenCV libraries](../../README.md#referencing-opencv-libraries).

## Parameters
* **Num Levels**:	Number of pyramid layers including the intial image.
* **Pyramid Scale**:	Image scale to build pyramid layers.
* **Window Size**:	Averaging window size.
* **Iterations**:	Number of iteration at each pyramid level.
* **Poly N**:	Size of the pixel neighborhood used to find polynomial expansion in each pixel.
* **Poly Sigma**:	Standard deviation of the Gaussian thta is used to smooth derivatives used as
	basis for the polynomial expansion.
* **Use Gaussian Filter**:	Uses the Gaussian Window Size x Window Size filter instead of a box filter.
* **Use Previous Flow**:	Use the optical flow of the previous frame as an estimate for the current frame.

This TOP takes one input where the optical flow of sequential frames is calculated.
