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

#include "ObjectDetectorTOP.h"
#include "Parameters.h"

#include <cassert>
#include <string>
#include <sstream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>

enum class
InfoChopChan
{
	Tracked,
	Confidence,
	Tx,
	Ty,
	W,
	H,
	Size
};

// These functions are basic C function, which the DLL loader can find
// much easier than finding a C++ Class.
// The DLLEXPORT prefix is needed so the compile exports these functions from the .dll
// you are creating
extern "C"
{

DLLEXPORT
void
FillTOPPluginInfo(TOP_PluginInfo *info)
{
	// This must always be set to this constant
	info->apiVersion = TOPCPlusPlusAPIVersion;

	// Change this to change the executeMode behavior of this plugin.
	info->executeMode = TOP_ExecuteMode::CPUMemWriteOnly;

	// For more information on OP_CustomOPInfo see CPlusPlus_Common.h
	OP_CustomOPInfo& customInfo = info->customOPInfo;

	// Unique name of the node which starts with an upper case letter, followed by lower case letters or numbers
	customInfo.opType->setString("Objectdetector");
	// English readable name
	customInfo.opLabel->setString("Object Detector");
	// Information of the author of the node
	customInfo.authorName->setString("Gabriel Robels");
	customInfo.authorEmail->setString("support@derivative.ca");

	// This TOP takes one input
	customInfo.minInputs = 1;
	customInfo.maxInputs = 1;
}

DLLEXPORT
TOP_CPlusPlusBase*
CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per TOP that is using the .dll
	return new ObjectDetectorTOP(info);
}

DLLEXPORT
void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context *context)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the TOP using that instance is deleted, or
	// if the TOP loads a different DLL
	delete (ObjectDetectorTOP*)instance;
}

};


ObjectDetectorTOP::ObjectDetectorTOP(const OP_NodeInfo* info) : 
	myFrame{ new cv::Mat() }, myClassifier{ new cv::CascadeClassifier() }, 
	myObjects{}, myLevelWeights{}, myRejectLevels{}, myPath{}, myScale{}, 
	myMinNeighbors{}, myLimitSize{}, myMinSize{}, myMaxSize{}, myDrawBoundingBox{}, 
	myLimitObjs{}, myMaxObjs{}
{
}

ObjectDetectorTOP::~ObjectDetectorTOP()
{
	delete myFrame;
	delete myClassifier;
}

void
ObjectDetectorTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	ginfo->cookEveryFrameIfAsked = false;
    ginfo->memPixelType = OP_CPUMemPixelType::BGRA8Fixed;
}

bool
ObjectDetectorTOP::getOutputFormat(TOP_OutputFormat* format, const OP_Inputs* inputs, void* reserved1)
{
	// In this function we could assign variable values to 'format' to specify
	// the pixel format/resolution etc that we want to output to.
	// If we did that, we'd want to return true to tell the TOP to use the settings we've
	// specified.
	// In this example we'll return false and use the input TOP's settings
	return false;
}


void
ObjectDetectorTOP::execute(TOP_OutputFormatSpecs* output,
						const OP_Inputs* inputs,
						TOP_Context *context,
						void* reserved1)
{
	using namespace cv;
	handleParameters(inputs);

	inputToMat(inputs);
	if (myFrame->empty())
		return;

	resize(*myFrame, *myFrame, cv::Size(output->width, output->height));

	Mat	frameGray;
	cvtColor(*myFrame, frameGray, COLOR_BGRA2GRAY);

	try
	{
		myClassifier->load(myPath);
		myClassifier->detectMultiScale(frameGray, myObjects, myRejectLevels, myLevelWeights, myScale, myMinNeighbors, 0, myMinSize, myMaxSize, true);
	}
	catch (...)
	{
		// If something went wrong just empty detected objects
		myObjects.clear();
	}

	if (myLimitObjs && myObjects.size() > myMaxObjs)
		myObjects.resize(myMaxObjs);

	if (myDrawBoundingBox)
		drawBoundingBoxes();

	cvMatToOutput(*myFrame, output);
}

void
ObjectDetectorTOP::setupParameters(OP_ParameterManager* manager, void*)
{
	myParms.setup(manager);
}

int32_t 
ObjectDetectorTOP::getNumInfoCHOPChans(void*)
{
	if (myLimitObjs)
		return static_cast<int32_t>(InfoChopChan::Size) * myMaxObjs + 1;
	else
		return static_cast<int32_t>(InfoChopChan::Size) * static_cast<int32_t>(myObjects.size()) + 1;
}

void 
ObjectDetectorTOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chop, void*)
{
	if (index == 0)
	{
		chop->name->setString("objects_tracked");
		chop->value = static_cast<float>(myObjects.size());
		return;
	}

	--index;	// Reduce one since the first channel is fixed

	int					obj = index / static_cast<int>(InfoChopChan::Size) + 1;
	int					prop = index % static_cast<int>(InfoChopChan::Size);
	bool				tracked = myObjects.size() >= obj;

	if (!tracked)
		chop->value = 0.0f;

	std::ostringstream	oss;
	oss << "obj" << obj << ":";

	switch (static_cast<InfoChopChan>(prop))
	{
		case InfoChopChan::Tracked:
		{
			oss << "tracked";
			if (tracked)
				chop->value = 1.0f;
			break;
		}
		case InfoChopChan::Confidence:
		{
			oss << "levelweight";
			if (tracked)
				chop->value = static_cast<float>(myLevelWeights.at(obj - 1));
			break;
		}
		case InfoChopChan::Tx:
		{
			oss << "tx";
			if (tracked)
				chop->value = static_cast<float>(myObjects.at(obj - 1).x);
			break;
		}
		case InfoChopChan::Ty:
		{
			oss << "ty";
			if (tracked)
				chop->value = static_cast<float>(myObjects.at(obj - 1).y);
			break;
		}
		case InfoChopChan::W:
		{
			oss << "w";
			if (tracked)
				chop->value = static_cast<float>(myObjects.at(obj - 1).width);
			break;
		}
		case InfoChopChan::H:
		{
			oss << "h";
			if (tracked)
				chop->value = static_cast<float>(myObjects.at(obj - 1).height);
			break;
		}
	}

	const std::string& tmp = oss.str();
	chop->name->setString(tmp.c_str());
}

bool 
ObjectDetectorTOP::getInfoDATSize(OP_InfoDATSize* info, void*)
{
	info->byColumn = false;
	info->cols = static_cast<int>(InfoChopChan::Size) + 1;
	info->rows = myLimitObjs ? myMaxObjs + 1 : static_cast<int32_t>(myObjects.size() + 1);
	return true;
}

void 
ObjectDetectorTOP::getInfoDATEntries(int32_t index, int32_t nEntries, OP_InfoDATEntries* entries, void*)
{
	if (index == 0)
	{
		entries->values[0]->setString("Object");
		entries->values[1]->setString("Tracked");
		entries->values[2]->setString("Level Weight");
		entries->values[3]->setString("Tx");
		entries->values[4]->setString("Ty");
		entries->values[5]->setString("W");
		entries->values[6]->setString("H");
	}
	else
	{
		std::ostringstream	oss;
		oss << "obj" << index;
		const std::string& tmp = oss.str();
		entries->values[0]->setString(tmp.c_str());
		if (index > myObjects.size())
		{
			entries->values[1]->setString("0");
			entries->values[2]->setString("0");
			entries->values[3]->setString("0");
			entries->values[4]->setString("0");
			entries->values[5]->setString("0");
			entries->values[6]->setString("0");
		}
		else
		{
			int obj = index - 1;
			char buffer[64];
			entries->values[1]->setString("1");\
			sprintf_s(buffer, "%f", myLevelWeights.at(obj));
			entries->values[2]->setString(buffer);
			sprintf_s(buffer, "%d", myObjects.at(obj).x);
			entries->values[3]->setString(buffer);
			sprintf_s(buffer, "%d", myObjects.at(obj).y);
			entries->values[4]->setString(buffer);
			sprintf_s(buffer, "%d", myObjects.at(obj).width);
			entries->values[5]->setString(buffer);
			sprintf_s(buffer, "%d", myObjects.at(obj).height);
			entries->values[6]->setString(buffer);
		}
	}
}

void 
ObjectDetectorTOP::handleParameters(const OP_Inputs* in)
{
	myPath = myParms.evalClassifier(in);
	myScale = myParms.evalScalefactor(in);
	myMinNeighbors = myParms.evalMinneighbors(in);

	myLimitSize = myParms.evalLimitobjectsize(in);
	in->enablePar(MaxobjectwidthName, myLimitSize);
	in->enablePar(MaxobjectheightName, myLimitSize);
	in->enablePar(MinobjectwidthName, myLimitSize);
	in->enablePar(MinobjectheightName, myLimitSize);

	if (myLimitSize)
	{
		const OP_TOPInput* top = in->getInputTOP(0);
		int32_t totalH = top->height;
		int32_t totalW = top->width;

		double	w, h;
		w = myParms.evalMinobjectwidth(in);
		h = myParms.evalMinobjectheight(in);
		myMinSize = cv::Size(static_cast<int>(w * totalW), static_cast<int>(h * totalH));
		w = myParms.evalMaxobjectwidth(in);
		h = myParms.evalMaxobjectheight(in);
		myMaxSize = cv::Size(static_cast<int>(w * totalW), static_cast<int>(h * totalH));
	}
	else
	{
		myMinSize = cv::Size();
		myMaxSize = cv::Size();
	}

	myDrawBoundingBox = myParms.evalDrawboundingbox(in);
	myLimitObjs = myParms.evalLimitobjectsdetected(in);
	in->enablePar(MaximumobjectsName, myLimitObjs);
	myMaxObjs = myLimitObjs ? myParms.evalMaximumobjects(in) : 0;
        myDownloadtype = static_cast<OP_TOPInputDownloadType>(myParms.evalDownloadtype(in));
}

void 
ObjectDetectorTOP::cvMatToOutput(const cv::Mat& M, TOP_OutputFormatSpecs* out) const
{
	size_t		height = out->height;
	size_t		width = out->width;

	out->newCPUPixelDataLocation = 0;
	uint8_t*	pixel = static_cast<uint8_t*>(out->cpuPixelData[0]);
	
	cv::flip(M, M, 0);
	uint8_t*	data = static_cast<uint8_t*>(M.data);

	memcpy(pixel, data, 4 * height * width * sizeof(uint8_t));
}

void 
ObjectDetectorTOP::inputToMat(const OP_Inputs* in) const
{
	const OP_TOPInput*	top = in->getInputTOP(0);
	if (!top)
		return;

	OP_TOPInputDownloadOptions	opts = {};
	opts.verticalFlip = true;
        opts.downloadType = myDownloadtype;

	uint8_t*	pixel = (uint8_t*)in->getTOPDataInCPUMemory(top, &opts);
	if (!pixel)
		return;

	int	height = top->height;
	int	width = top->width;

	*myFrame = cv::Mat(height, width, CV_8UC4);
	uint8_t*	data = (uint8_t*)myFrame->data;

	memcpy(data, pixel, 4 * height * width * sizeof(uint8_t));
}

void
ObjectDetectorTOP::drawBoundingBoxes() const
{
	cv::Scalar color = cv::Scalar(255, 0, 0);
	for (const cv::Rect& obj : myObjects)
	{
		rectangle(*myFrame, obj, color, 2);
	}
}
