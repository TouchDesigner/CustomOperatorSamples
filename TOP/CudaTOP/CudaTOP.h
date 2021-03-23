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
#ifndef __CudaTOP__
#define __CudaTOP__

#include "TOP_CPlusPlusBase.h"
#include "Parameters.h"

#include <string>
#include <array>

/*
This example implements a TOP to copy a texture or output a plain color using CUDA.
It takes the following parameters:
	- Color:	The color to output if no input is detected.
*/

// Check methods [getNumInfoCHOPChans, getInfoCHOPChan, getInfoDATSize, getInfoDATEntries]
// if you want to output values to the Info CHOP/DAT

// To get more help about these functions, look at TOP_CPlusPlusBase.h
class CudaTOP : public TOP_CPlusPlusBase
{
public:
    CudaTOP(const OP_NodeInfo *info);
    virtual ~CudaTOP();

    virtual void		getGeneralInfo(TOP_GeneralInfo*, const OP_Inputs*, void* reserved) override;

    virtual bool		getOutputFormat(TOP_OutputFormat*, const OP_Inputs*, void* reserved) override;

    virtual void		execute(TOP_OutputFormatSpecs*, const OP_Inputs*, TOP_Context*, void* reserved) override;

	virtual void		setupParameters(OP_ParameterManager*, void* reserved) override;

	virtual void		getErrorString(OP_String*, void* reserved) override;

private:
	void		checkOutputFormat(const TOP_OutputFormatSpecs*);

	void		checkTopFormat(const OP_TOPInput*, const TOP_OutputFormatSpecs*);

	std::string	myError;

	// Parameters
	std::array<uint8_t, 4>	myRgba8;

	Parameters myParms;
};

#endif
