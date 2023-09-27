# Object Detector TOP

This example implements a TOP to detect objects using OpenCV's Cascade Classifier. For
more information on the parameters check cv::CascadeClassifier documentation.

## Prerequisites
Requires a [reference](https://github.com/TouchDesigner/CustomOperatorSamples#referencing-opencv-libraries) to the openCV include and library folder.

## Parameters
* **Classifier**:   A path to a .xml pretrained classifier. It can be either Haar or 
		LBP. OpenCV includes pretrained classiffiers and can be found in 
		opencv/sources/data.
* **Scale Factor**: Specifies how much the image size is reduced at each image scale.
* **Min Neighbors**:    How many neighbors each candidate rectangle should have to retain it.
* **Limit Object Size**:    If on, limit the size of the detected objects.
* **Min Object Size**:  Minimum possible object size. Objects smaller than that are ignored.
* **Max Object Size**:  Maximum possible object size. Objects larger than that are ignored. If 
	maxSize == minSize model is evaluated on single scale.
* **Draw Bounding Box**:    If on, draw the bounding box of the detected object.
* **Limit Objects Detected**:   If on, limit the number of objects detected. Turn this parameter on
	if you need the channels outputted to CHOPInfo to be constant.
* **Maximum Objects**:  The maximum number of objects that the TOP can detects

This TOP takes one input where to detect faces. Outputs the input data with the bounding boxes for the 
detected objects.

## Info CHOP and Info DAT outputs: 
* **objects_tracked**:  Number of objects that we are currently tracking.
* **obj#:tracked**: Whether this channel group is tracking an object.
* **obj#:levelweight**: The certainty of classification at the final stage. This value can then be used 
	to separate strong from weaker classifications.
* **obj#:tx**:  X position of the bounding box.
* **obj#:ty**:  Y position of the bounding box.
* **obj#:w**:   Width of the bounding box.
* **obj#:h**:   Height of the bounding box.