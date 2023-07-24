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

#ifndef __BasicGeneratorCHOP__
#define __BasicGeneratorCHOP__

#include "CHOP_CPlusPlusBase.h"

using namespace TD;

/*
This example implements a CHOP which takes the following parameters:
	- Length: The number of samples produced per channel.
	- Number of Channels: The number of channels produced.
	- Apply Scale: If On, scale values.
	- Scale: A scalar by which the output signal is scaled.
	- Operation: One of [Add, Multiply, Power] which controls which operation the output signal constitutes

The output values are: scale*(channel operation sample)

This CHOP is a generator so it does not need an input and it is not time sliced.
*/

// Check methods [getNumInfoCHOPChans, getInfoCHOPChan, getInfoDATSize, getInfoDATEntries]
// if you want to output values to the Info CHOP/DAT

// To get more help about these functions, look at CHOP_CPlusPlusBase.h
class BasicGeneratorCHOP : public CHOP_CPlusPlusBase
{
public:
	BasicGeneratorCHOP(const OP_NodeInfo* info);
	virtual ~BasicGeneratorCHOP();

	virtual void		getGeneralInfo(CHOP_GeneralInfo*, const TD::OP_Inputs*, void*) override;
	virtual bool		getOutputInfo(CHOP_OutputInfo*, const TD::OP_Inputs*, void*) override;
	virtual void		getChannelName(int32_t index, OP_String *name, const TD::OP_Inputs*, void*) override;

	virtual void		execute(CHOP_Output*, const TD::OP_Inputs*, void*) override;

	virtual void		setupParameters(TD::OP_ParameterManager* manager, void*) override;

private:
};

#endif
