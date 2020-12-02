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

#ifndef __WrapPointsSOP__
#define __WrapPointsSOP__

#include "SOP_CPlusPlusBase.h"

#include <string>

struct Parameters;

/*
This example implements a SOP which uses sendRay to wrap input 1 to input 2.
	- Rays:	One of [Parallel, Radial], which determines in which way the rays are sent.
	- Direction: If Rays is Parallel, it will send rays from the points of input 1 in this vector's direction.
	- Destination: If Rays is Radial, it will send rays from the points of input 1 towards this destination.
	- Reverse: If On, reverses the direction in which the rays are sent.
	- Hit Color: Which color to set those points that hit input 2.
	- Miss Color: Which color to set those points that did not hit input 2.
	- Scale: A scale for hitLength, 0 a hit point will stay in its original position; 1 a hit point will be at the location
		where it hit the second input.

This SOP takes two inputs and wraps the first one onto the second one.
*/

// Check methods [getNumInfoCHOPChans, getInfoCHOPChan, getInfoDATSize, getInfoDATEntries]
// if you want to output values to the Info CHOP/DAT

// To get more help about these functions, look at SOP_CPlusPlusBase.h
class WrapPointsSOP : public SOP_CPlusPlusBase
{
public:
	WrapPointsSOP(const OP_NodeInfo* info);
	virtual ~WrapPointsSOP();

	virtual void		getGeneralInfo(SOP_GeneralInfo*, const OP_Inputs*, void*) override;

	virtual void		execute(SOP_Output*, const OP_Inputs*, void*) override;

	virtual void		executeVBO(SOP_VBOOutput*, const OP_Inputs*, void*) override;

	virtual void		setupParameters(OP_ParameterManager*, void*) override;

	virtual void		getWarningString(OP_String*, void*) override;

private:
	void		castParallel(SOP_Output*, const OP_SOPInput*, const OP_SOPInput*, Vector);

	void		castRadial(SOP_Output*, const OP_SOPInput*, const OP_SOPInput*, Position);

	// This method returns true if the input geometry only has triangles and lines
	void		copyPrimitives(SOP_Output*, const OP_SOPInput*);

	// Before calling this functions SOP_Output should contain as many points as OP_SOPInput
	void		copyAttributes(SOP_Output*, const OP_SOPInput*) const;

	void		copyColors(SOP_Output*, const OP_SOPInput*) const;

	void		copyTextures(SOP_Output*, const OP_SOPInput*) const;

	void		copyCustomAttributes(SOP_Output*, const OP_SOPInput*) const;

	void		castPoint(SOP_Output*, const Position*, const Vector* normals, int index, const OP_SOPInput* geo, Vector direction);

	std::string	myWarningString;

	Parameters*	myParms;
};

#endif // !__WrapPointsSOP__
