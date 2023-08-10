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
#ifndef __BasicFilterTOP__
#define __BasicFilterTOP__

#include "TOP_CPlusPlusBase.h"

#include <thread>
#include <condition_variable>
#include <atomic>
#include <array>
#include <queue>

class ThreadManager;

// for multithreading
const int NumCPUPixelDatas = 3;

using namespace TD;

/*
This example implements a TOP to limit the number of colors from the input. This example
executes on CPU Memory and supports single threaded and multi threaded.
It takes the following parameters:
	- Bits per Color:	The number of bits for the RGB channels. Therefore, if
		we set this parameter to 1. We limit our color palette to 2^(3*1) = 8 colors.
	- Dither:	If on, we apply a dithering algorithm to diffuse the error.
	- Multithreaded: If on, we calculate the output for 3 frames at the same time, therefore 
		it lags from the input by 3/4 frames depending on Download Type.
*/

// Check methods [getNumInfoCHOPChans, getInfoCHOPChan, getInfoDATSize, getInfoDATEntries]
// if you want to output values to the Info CHOP/DAT

// To get more help about these functions, look at TOP_CPlusPlusBase.h
class BasicFilterTOP : public TOP_CPlusPlusBase
{
public:
    BasicFilterTOP(const OP_NodeInfo *info, TOP_Context* context);
    virtual ~BasicFilterTOP();

    virtual void		getGeneralInfo(TOP_GeneralInfo*, const TD::OP_Inputs*, void* reserved) override;

    virtual void		execute(TOP_Output*, const TD::OP_Inputs*, void* reserved) override;

	virtual void		setupParameters(TD::OP_ParameterManager*, void* reserved) override;

private:
	void		switchToSingleThreaded();

	void		switchToMultiThreaded();

	TOP_Context* myContext;
	OP_SmartRef<OP_TOPDownloadResult> myPrevDownRes;

	// Threading variables
	std::array<ThreadManager*, NumCPUPixelDatas>	myThreadManagers;
	std::queue<ThreadManager*>						myThreadQueue;
	int												myExecuteCount;
	bool											myMultiThreaded;
};

#endif
