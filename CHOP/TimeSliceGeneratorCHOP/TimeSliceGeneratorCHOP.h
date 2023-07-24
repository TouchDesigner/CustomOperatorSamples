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

#ifndef __TimeSliceGeneratorCHOP__
#define __TimeSliceGeneratorCHOP__

#include "CHOP_CPlusPlusBase.h"

using namespace TD;

/*
This example implements a CHOP which takes the following parameters:
	- Type:	One of [Sine, Square, Ramp] which controls which wave we output.
	- Frequency: Determines the frequency of our wave.
	- Apply Scale: If On, scale values.
	- Scale: A scalar by which the output signal is scaled.

This CHOP is a generator so it does not need an input.

The output signal is: scale*(shape value at current time). Note that this CHOP is 
time sliced; therefore, we need to keep track of the current time to output the correct
value.
*/

// Check methods [getNumInfoCHOPChans, getInfoCHOPChan, getInfoDATSize, getInfoDATEntries]
// if you want to output values to the Info CHOP/DAT

// To get more help about these functions, look at CHOP_CPlusPlusBase.h
class TimeSliceGeneratorCHOP : public CHOP_CPlusPlusBase
{
public:
	TimeSliceGeneratorCHOP(const OP_NodeInfo* info);
	virtual ~TimeSliceGeneratorCHOP();

	virtual void		getGeneralInfo(CHOP_GeneralInfo*, const TD::OP_Inputs*, void*) override;
	virtual bool		getOutputInfo(CHOP_OutputInfo*, const TD::OP_Inputs*, void*) override;
	virtual void		getChannelName(int32_t index, OP_String *name, const TD::OP_Inputs*, void*) override;

	virtual void		execute(CHOP_Output*, const TD::OP_Inputs*, void*) override;

	virtual void		setupParameters(TD::OP_ParameterManager* manager, void*) override;

private:
	double myOffset;
};

#endif
