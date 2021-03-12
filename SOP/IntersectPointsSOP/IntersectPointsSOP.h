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
class IntersectPointsSOP : public SOP_CPlusPlusBase
{
public:
	IntersectPointsSOP(const OP_NodeInfo* info);
	virtual ~IntersectPointsSOP();

	virtual void		getGeneralInfo(SOP_GeneralInfo*, const OP_Inputs*, void*) override;

	virtual void		execute(SOP_Output*, const OP_Inputs*, void*) override;

	virtual void		executeVBO(SOP_VBOOutput*, const OP_Inputs*, void*) override;

	virtual void		setupParameters(OP_ParameterManager*, void*) override;

	virtual void		getWarningString(OP_String*, void*) override;

	Parameters myParms;

private:
	void		copyPoints(SOP_Output*, const OP_SOPInput*) const;

	// Before calling this functions SOP_Output should contain as many points as OP_SOPInput
	void		copyAttributes(SOP_Output*, const OP_SOPInput*) const;

	// This method returns true if the input geometry only has triangles and lines
	void		copyPrimitives(SOP_Output*, const OP_SOPInput*);

	void		copyNormals(SOP_Output*, const OP_SOPInput*) const;

	void		copyColors(SOP_Output*, const OP_SOPInput*) const;

	void		copyTextures(SOP_Output*, const OP_SOPInput*) const;

	void		copyCustomAttributes(SOP_Output*, const OP_SOPInput*) const;

	std::string	myWarningString;

};

#endif // !__IntersectPointsSOP__
