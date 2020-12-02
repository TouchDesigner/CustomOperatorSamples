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

#ifndef __FilterSOP__
#define __FilterSOP__

#include "SOP_CPlusPlusBase.h"
#include <string>

/*
This example implements a SOP which takes the following parameters:
	- Translate CHOP: A CHOP with 3 channels whose value determines a translation for the input SOP.

This SOP is a filter and it takes one input SOP with triangulated geometry.
*/

// Check methods [getNumInfoCHOPChans, getInfoCHOPChan, getInfoDATSize, getInfoDATEntries]
// if you want to output values to the Info CHOP/DAT

// To get more help about these functions, look at SOP_CPlusPlusBase.h
class FilterSOP : public SOP_CPlusPlusBase
{
public:
	FilterSOP(const OP_NodeInfo* info);
	virtual ~FilterSOP();

	virtual void		getGeneralInfo(SOP_GeneralInfo*, const OP_Inputs*, void*) override;

	virtual void		execute(SOP_Output*, const OP_Inputs*, void*) override;

	virtual void		executeVBO(SOP_VBOOutput*, const OP_Inputs*, void*) override;

	virtual void		setupParameters(OP_ParameterManager* manager, void*) override;

	virtual void		getWarningString(OP_String*, void*) override;

private:
	void		copyPointsTranslated(SOP_Output*, const OP_SOPInput*, const Vector&) const;

	// Before calling this functions SOP_Output should contain as many points as OP_SOPInput
	void		copyAttributes(SOP_Output*, const OP_SOPInput*) const;
	
	void		copyNormals(SOP_Output*, const OP_SOPInput*) const;

	void		copyColors(SOP_Output*, const OP_SOPInput*) const;

	void		copyTextures(SOP_Output*, const OP_SOPInput*) const;

	void		copyCustomAttributes(SOP_Output*, const OP_SOPInput*) const;
	// This method returns true if the input geometry only has triangles and lines
	void		copyPrimitives(SOP_Output*, const OP_SOPInput*);

	Vector		getTranslate(const OP_CHOPInput*);

	std::string			myWarningString;
};

#endif // !__FilterSOP__
