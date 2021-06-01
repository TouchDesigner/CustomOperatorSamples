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

class ThreadManager;
class Parameters;

/*
This example implements a TOP to limit the number of colors from the input. This example
executes on CPU Memory and supports single threaded and multi threaded.
It takes the following parameters:
	- Bits per Color:	The number of bits for the RGB channels. Therefore, if
		we set this parameter to 1. We limit our color palette to 2^(3*1) = 8 colors.
	- Dither:	If on, we apply a dithering algorithm to diffuse the error.
	- Multithreaded: If on, we calculate the output for 3 frames at the same time, therefore 
		it lags from the input by 3/4 frames depending on Download Type.
	- Download Type: One of [ Instant, Delayed ]. Specifies if the data is downloaded to the CPUMem
		this frame or next frame.
*/

// Check methods [getNumInfoCHOPChans, getInfoCHOPChan, getInfoDATSize, getInfoDATEntries]
// if you want to output values to the Info CHOP/DAT

// To get more help about these functions, look at TOP_CPlusPlusBase.h
class BasicFilterTOP : public TOP_CPlusPlusBase
{
public:
    BasicFilterTOP(const OP_NodeInfo *info);
    virtual ~BasicFilterTOP();

    virtual void		getGeneralInfo(TOP_GeneralInfo*, const OP_Inputs*, void* reserved) override;

    virtual bool		getOutputFormat(TOP_OutputFormat*, const OP_Inputs*, void* reserved) override;

    virtual void		execute(TOP_OutputFormatSpecs*, const OP_Inputs*, TOP_Context*, void* reserved) override;

	virtual void		setupParameters(OP_ParameterManager*, void* reserved) override;

private:
	void		switchToSingleThreaded();

	void		switchToMultiThreaded();

	Parameters*	myParms;

	// Threading variables
	std::array<ThreadManager*, NumCPUPixelDatas>	myThreadManagers;
	int												myExecuteCount;
	bool											myMultiThreaded;
};

#endif
