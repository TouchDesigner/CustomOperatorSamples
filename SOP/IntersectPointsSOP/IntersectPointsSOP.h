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

#ifndef __IntersectPointsSOP__
#define __IntersectPointsSOP__

#include "SOP_CPlusPlusBase.h"
#include "Parameters.h"

#include <string>

/*
This example implements a SOP outputs the point of the first input colored
whether they are inside or outside the second input. The output points have an integer attribute
Inside. 1 if the point is inside, 0 otherwise.
	- Inside Value:	Color to give to the points from the first input that are inside
		the geometry of the second input.
	- Outside Value:	Color to give to the points from the first input that are outside
		the geometry of the second input.

This SOP takes two inputs:
	- First: Points to be colored.
	- Second: Geometry to test whether points are inside or not.
*/

// Check methods [getNumInfoCHOPChans, getInfoCHOPChan, getInfoDATSize, getInfoDATEntries]
// if you want to output values to the Info CHOP/DAT

// To get more help about these functions, look at SOP_CPlusPlusBase.h
class IntersectPointsSOP : public TD::SOP_CPlusPlusBase
{
public:
	IntersectPointsSOP(const TD::OP_NodeInfo* info);
	virtual ~IntersectPointsSOP();

	virtual void		getGeneralInfo(TD::SOP_GeneralInfo*, const TD::OP_Inputs*, void*) override;

	virtual void		execute(TD::SOP_Output*, const TD::OP_Inputs*, void*) override;

	virtual void		executeVBO(TD::SOP_VBOOutput*, const TD::OP_Inputs*, void*) override;

	virtual void		setupParameters(TD::OP_ParameterManager*, void*) override;

	virtual void		getWarningString(TD::OP_String*, void*) override;

private:
	void		copyPoints(TD::SOP_Output*, const TD::OP_SOPInput*) const;

	// Before calling this functions SOP_Output should contain as many points as OP_SOPInput
	void		copyAttributes(TD::SOP_Output*, const TD::OP_SOPInput*) const;

	// This method returns true if the input geometry only has triangles and lines
	void		copyPrimitives(TD::SOP_Output*, const TD::OP_SOPInput*);

	void		copyNormals(TD::SOP_Output*, const TD::OP_SOPInput*) const;

	void		copyColors(TD::SOP_Output*, const TD::OP_SOPInput*) const;

	void		copyTextures(TD::SOP_Output*, const TD::OP_SOPInput*) const;

	void		copyCustomAttributes(TD::SOP_Output*, const TD::OP_SOPInput*) const;

	std::string	myWarningString;
	
	Parameters myParms;
};

#endif // !__IntersectPointsSOP__
