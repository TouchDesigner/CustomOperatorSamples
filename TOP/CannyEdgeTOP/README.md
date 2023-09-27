# Canny Edge TOP

This example implements a TOP exposing the canny edge detector using openCV's cuda functionallity.

For more information visit: https://docs.opencv.org/3.4/d0/d05/group__cudaimgproc.html#gabc17953de36faa404acb07dc587451fc

## Prerequisites
Requires a [reference](https://github.com/TouchDesigner/CustomOperatorSamples#referencing-opencv-libraries) to the openCV include and library folder.

## Parameters
* **Low Threshold**: Minimum value for the intensity gradient to decide if it is used as a edge.
* **High Threshold**: Maximum value for the intensity gradient to decide if it is used as a edge.
* **Apperture Size**: Kernel size for the Sobel operator.
* **L2 Gradient**: ndicating whether a more accurate L2 should be used to calculate the image gradient magnitude.

This TOP takes one input which must be 8 bit single channel.



