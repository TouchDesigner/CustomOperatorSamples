# Canny Edge TOP

## Prerequisites
Requires a [reference](https://github.com/TouchDesigner/CustomOperatorSamples#referencing-opencv-libraries) to the openCV include and library folder.

## Parameters
* **Low Threshold**: Minimum value for the intensity gradient to decide if it is used as a edge.
* **High Threshold**: Maximum value for the intensity gradient to decide if it is used as a edge.
* **Apperture Size**: Kernel size for the Sobel operator.
* **L2 Gradient**: ndicating whether a more accurate L2 should be used to calculate the image gradient magnitude.
