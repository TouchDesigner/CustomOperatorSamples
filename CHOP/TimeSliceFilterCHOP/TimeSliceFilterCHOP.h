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

#ifndef __TimeSliceFilterCHOP__
#define __TimeSliceFilterCHOP__

#include "CHOP_CPlusPlusBase.h"
#include "Parameters.h"
#include <vector>

class FilterValues;

/*
This example implements a CHOP which takes the following parameters:
	- Operation: One of [Max, Min, Average] which controls which operation is applied to
					the input signal.
	- Reset: A pulse to reset the signal.

This CHOP is a filter and it takes at least one input.

The output signal is: the current maximum, minimum, or average value of the input signal
*/

// Check methods [getNumInfoCHOPChans, getInfoCHOPChan, getInfoDATSize, getInfoDATEntries]
// if you want to output values to the Info CHOP/DAT

// To get more help about these functions, look at CHOP_CPlusPlusBase.h
class TimeSliceFilterCHOP : public CHOP_CPlusPlusBase
{
public:
	TimeSliceFilterCHOP(const OP_NodeInfo* info);
	virtual ~TimeSliceFilterCHOP();

	virtual void		getGeneralInfo(CHOP_GeneralInfo*, const OP_Inputs*, void*) override;
	virtual bool		getOutputInfo(CHOP_OutputInfo*, const OP_Inputs*, void*) override;
	virtual void		getChannelName(int32_t index, OP_String *name, const OP_Inputs*, void*) override;

	virtual void		execute(CHOP_Output*, const OP_Inputs*, void*) override;

	virtual void		setupParameters(OP_ParameterManager* manager, void*) override;
	virtual void		pulsePressed(const char* name, void* reserved1) override;

private:
	Parameters myParms;
	std::vector<FilterValues>	myValues;
};

#endif
