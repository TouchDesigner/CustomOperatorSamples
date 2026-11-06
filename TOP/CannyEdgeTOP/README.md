# Canny Edge TOP

This example implements a TOP exposing the canny edge detector using OpenCV's cuda functionallity.

For more information visit: https://docs.opencv.org/3.4/d0/d05/group__cudaimgproc.html#gabc17953de36faa404acb07dc587451fc

## Prerequisites
Uses the OpenCV that ships with TouchDesigner. CMake finds it in the TouchDesigner installation set by `TD_INSTALLATION_PATH`, see [Referencing OpenCV libraries](../../README.md#referencing-opencv-libraries).
This example uses OpenCV 4.11, bundled by default with TouchDesigner 2025.30000 builds.

Requires an Nvidia GPU. Currently this example uses CUDA as the execution engine.
Requires CUDA 12.8 Toolkit to be [installed](../../README.md#installing-the-cuda-toolkit-windows-only).
## Parameters
* **Low Threshold**: Minimum value for the intensity gradient to decide if it is used as a edge.
* **High Threshold**: Maximum value for the intensity gradient to decide if it is used as a edge.
* **Apperture Size**: Kernel size for the Sobel operator.
* **L2 Gradient**: ndicating whether a more accurate L2 should be used to calculate the image gradient magnitude.

This TOP takes one input which must be 8 bit single channel.
