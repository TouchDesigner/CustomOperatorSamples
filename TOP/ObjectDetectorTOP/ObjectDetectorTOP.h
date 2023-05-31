/* Shared Use License: This file is owned by Derivative Inc. (Derivative)
* and can only be used, and/or modified for use, in conjunction with
* Derivative's TouchDesigner software, and only if you are a licensee who has
* accepted Derivative's TouchDesigner license or assignment agreement
* (which also govern the use of this file). You may share or redistribute
* a modified version of this file provided the following conditions are met:
*
* 1. The shared file or redistribution must retain the information set out
* above and this list of conditions.
* 2. Derivative's name (Derivative Inc.) or its trademarks may not be used
* to endorse or promote products derived from this file without specific
* prior written permission from Derivative.
*/
#ifndef __ObjectDetectorTOP__
#define __ObjectDetectorTOP__

#include "TOP_CPlusPlusBase.h"

#include <vector>
#include <string>
#include <opencv2/core.hpp>

namespace cv
{
    class Mat;
    class CascadeClassifier;
}

/*
This example implements a TOP to detect objects using OpenCV's Cascade Classifier. For
more information on the parameters check cv::CascadeClassifier documentation.
It takes the following parameters:
	- Classifier:   A path to a .xml pretrained classifier. It can be either Haar or 
		LBP. OpenCV includes pretrained classiffiers and can be found in 
		opencv/sources/data.
	- Scale Factor: Specifies how much the image size is reduced at each image scale.
	- Min Neighbors:    How many neighbors each candidate rectangle should have to retain it.
	- Limit Object Size:    If on, limit the size of the detected objects.
	- Min Object Size:  Minimum possible object size. Objects smaller than that are ignored.
	- Max Object Size:  Maximum possible object size. Objects larger than that are ignored. If 
		maxSize == minSize model is evaluated on single scale.
	- Draw Bounding Box:    If on, draw the bounding box of the detected object.
	- Limit Objects Detected:   If on, limit the number of objects detected. Turn this parameter on
		if you need the channels outputted to CHOPInfo to be constant.
	- Maximum Objects:  The maximum number of objects that the TOP can detects
        - Download Type:    How the input texture is downloaded.

This TOP takes one input where to detect faces. Outputs the input data with the bounding boxes for the 
detected objects. It outputs the following information to CHOPInfo and DATInfo: 
	- objects_tracked:  Number of objects that we are currently tracking.
	- obj#:tracked: Whether this channel group is tracking an object.
	- obj#:levelweight: The certainty of classification at the final stage. This value can then be used 
		to separate strong from weaker classifications.
	- obj#:tx:  X position of the bounding box.
	- obj#:ty:  Y position of the bounding box.
	- obj#:w:   Width of the bounding box.
	- obj#:h:   Height of the bounding box.

Note that it will take two cooks to get be able to see the output of an inputted frame - as the output is delayed by one frame
*/

enum class OP_TOPInputDownloadType;

// To get more help about these functions, look at TOP_CPlusPlusBase.h
class ObjectDetectorTOP : public TD::TOP_CPlusPlusBase
{
public:
    ObjectDetectorTOP(const TD::OP_NodeInfo *info, TD::TOP_Context* context);
    virtual ~ObjectDetectorTOP();

    virtual void		getGeneralInfo(TD::TOP_GeneralInfo*, const TD::OP_Inputs*, void* reserved1) override;

    virtual void		execute(TD::TOP_Output*, const TD::OP_Inputs*, void* reserved) override;

	virtual void		setupParameters(TD::OP_ParameterManager *manager, void *reserved1) override;

    virtual int32_t     getNumInfoCHOPChans(void*) override;

    virtual void        getInfoCHOPChan(int32_t, TD::OP_InfoCHOPChan*, void*) override;

    virtual bool        getInfoDATSize(TD::OP_InfoDATSize*, void*) override;

    virtual void        getInfoDATEntries(int32_t, int32_t, TD::OP_InfoDATEntries*, void*) override;

private:
    void                handleParameters(const TD::OP_Inputs*);

    void                cvMatToOutput(const cv::Mat&, TD::TOP_Output*, TD::TOP_UploadInfo info) const;

    void                inputToMat(const TD::OP_Inputs*);

    void                drawBoundingBoxes() const;

    cv::Mat*                myFrame;
    cv::CascadeClassifier*  myClassifier;
    std::vector<cv::Rect>   myObjects;
    std::vector<double>     myLevelWeights;
    std::vector<int>        myRejectLevels;

    // Parameters
    std::string myPath;
    double      myScale;
    int         myMinNeighbors;
    bool        myLimitSize;
    cv::Size    myMinSize;
    cv::Size    myMaxSize;
    bool        myDrawBoundingBox;
    bool        myLimitObjs;
    int         myMaxObjs;

	int					myExecuteCount;
	TD::TOP_Context* myContext;
	TD::OP_SmartRef<TD::OP_TOPDownloadResult> myPrevDownRes;
};

#endif
