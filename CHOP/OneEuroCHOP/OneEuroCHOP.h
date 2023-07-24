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
#ifndef __OneEuroCHOP__
#define __OneEuroCHOP__

#include "CHOP_CPlusPlusBase.h"
#include <vector>

class OneEuroImpl;

using namespace TD;

/*
This example implements a CHOP that filters noice out of a signl. 
It takes the following parameters:
	- Cutoff Frequency (Hz): Decrease it if slow speed jitter is a problem.
	- Speed Coefficient: Increase if high speed lag is a problem.
	- Slope Cutoff Frequency (Hz): Avoids high derivative bursts caused by jitter.

References: 
	- Casiez, G., Roussel, N. and Vogel, D. (2012). 1€ Filter: A Simple 
	Speed-based Low-pass Filter for Noisy Input in Interactive Systems. 
	Proceedings of the ACM Conference on Human Factors in Computing Systems 
	(CHI '12). Austin, Texas (May 5-12, 2012). New York: ACM Press, pp. 2527-2530.

For more information about tuning the parameters check the paper mentioned.
This CHOP is a filter and it takes exactly one input.
*/

// Check methods [getNumInfoCHOPChans, getInfoCHOPChan, getInfoDATSize, getInfoDATEntries]
// if you want to output values to the Info CHOP/DAT

// To get more help about these functions, look at CHOP_CPlusPlusBase.h
class OneEuroCHOP : public CHOP_CPlusPlusBase
{
public:
	OneEuroCHOP(const OP_NodeInfo* info);
	virtual ~OneEuroCHOP();

	virtual void		getGeneralInfo(CHOP_GeneralInfo*, const TD::OP_Inputs*, void*) override;
	virtual bool		getOutputInfo(CHOP_OutputInfo*, const TD::OP_Inputs*, void*) override;
	virtual void		getChannelName(int32_t index, OP_String *name, const TD::OP_Inputs*, void*) override;

	virtual void		execute(CHOP_Output*, const TD::OP_Inputs*, void*) override;

	virtual void		setupParameters(TD::OP_ParameterManager* manager, void*) override;


private:
	void				handleParameters(const TD::OP_Inputs*, const OP_CHOPInput*);

	std::vector<OneEuroImpl*>	myFiltersPerChannel;
};

#endif
