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
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#include <string>

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
class WrapPointsSOP : public TD::SOP_CPlusPlusBase
{
public:
	WrapPointsSOP(const TD::OP_NodeInfo* info);
	virtual ~WrapPointsSOP();

	virtual void		getGeneralInfo(TD::SOP_GeneralInfo*, const TD::OP_Inputs*, void*) override;

	virtual void		execute(TD::SOP_Output*, const TD::OP_Inputs*, void*) override;

	virtual void		executeVBO(TD::SOP_VBOOutput*, const TD::OP_Inputs*, void*) override;

	virtual void		setupParameters(TD::OP_ParameterManager*, void*) override;

	virtual void		getWarningString(TD::OP_String*, void*) override;

private:
	void		castParallel(TD::SOP_Output*, const TD::OP_SOPInput*, const TD::OP_SOPInput*, TD::Vector, bool, double, TD::Color, TD::Color);

	void		castRadial(TD::SOP_Output*, const TD::OP_SOPInput*, const TD::OP_SOPInput*, TD::Position, bool, double, TD::Color, TD::Color);

	// This method returns true if the input geometry only has triangles and lines
	void		copyPrimitives(TD::SOP_Output*, const TD::OP_SOPInput*);

	// Before calling this functions SOP_Output should contain as many points as OP_SOPInput
	void		copyAttributes(TD::SOP_Output*, const TD::OP_SOPInput*) const;

	void		copyColors(TD::SOP_Output*, const TD::OP_SOPInput*) const;

	void		copyTextures(TD::SOP_Output*, const TD::OP_SOPInput*) const;

	void		copyCustomAttributes(TD::SOP_Output*, const TD::OP_SOPInput*) const;

	void		castPoint(TD::SOP_Output*, const TD::Position*, const TD::Vector* normals, int index, const TD::OP_SOPInput* geo, TD::Vector direction, double, TD::Color, TD::Color);

	std::string	myWarningString;
	
	Parameters	myParms;
};

#endif // !__WrapPointsSOP__
