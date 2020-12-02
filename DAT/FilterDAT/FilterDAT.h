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

#ifndef __DATFilter__
#define __DATFilter__

#include "DAT_CPlusPlusBase.h"

enum class Case;

/*
This example implements a DAT that takes one input and changes the content's case.

It takes the following parameters:
	- Case:	One of [Upper Camel Case, Lower Case, Upper Case]. Which determines how the 
		content's case changes.
	- Keep Spaces:	If On, the output will have white space.
*/

// Check methods [getNumInfoCHOPChans, getInfoCHOPChan, getInfoDATSize, getInfoDATEntries]
// if you want to output values to the Info CHOP/DAT

// To get more help about these functions, look at SOP_CPlusPlusBase.h

class DATFilter : public DAT_CPlusPlusBase
{
public:
	DATFilter(const OP_NodeInfo*);
	virtual ~DATFilter();

	virtual void		getGeneralInfo(DAT_GeneralInfo*, const OP_Inputs*, void* reserved) override;

	virtual void		execute(DAT_Output*, const OP_Inputs*, void* reserved) override;

	virtual void		setupParameters(OP_ParameterManager*, void* reserved) override;

private:
	void				handleParameters(const OP_Inputs*);

	void				fillTable(DAT_Output*, const OP_DATInput*);

	// Parameters
	Case	myCase;
	bool	myKeepSpaces;
};

#endif
