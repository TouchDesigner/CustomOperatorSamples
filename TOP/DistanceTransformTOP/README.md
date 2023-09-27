# Distance Transform TOP
This example implements a TOP to calculate the distance transform using openCV.

For more information visit: https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga8a0b7fdfcb7a13dde018988ba3a43042

## Prerequisites
Requires a [reference](https://github.com/TouchDesigner/CustomOperatorSamples#referencing-opencv-libraries) to the openCV include and library folder.

## Parameters
* **Distance Type**:        One of [L1, L2, C], which determines how to calculate the distance.
* **Mask Size**:        One of [3x3, 5x5, Precise], which determines the size of the transform mask.
* **Normalize**:	If On, normalize the output image.

This TOP takes one input which must be 8 bit single channel.