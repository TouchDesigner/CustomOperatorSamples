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

#ifndef __SprinkleSOP__
#define __SprinkleSOP__

#include "SOP_CPlusPlusBase.h"

#include <random>
#include <string>
#include <vector>
#include <array>

struct Parameters;

class RandomPointsBuffer;
class VolSprinkleTree;

/*
This example implements a SOP which outputs random points given a mesh and the following
parameters:
	- Seed:	A number to start the random number generator.
	- Generate:	One of [Surface Area, Per Primitive, Bounding Box, Inside Volume] which
		determines how to generate the points. Surface Area will generate points evenly on the 
		surface of the mesh. Per Primitive, will generate points evenly distributed among primitives. 
		Inside Bounding Box, will generate inside the bounding box of the geometry. Inside Volume, will 
		generate points inside the geometry.
	- Point Count:	The total number of points to generate.
	- Separate Points:	If On, it will separate the points so the distance between the points
		is at least the parameter Minimum Distance.
	- Minimum Distance:	The distance between each random point generated.

This SOP takes one to two inputs:
	-	The reference geometry where the SOP will generate points.
	-	If Generate is [Surface Area, Per Primitive], the points generated along input 1 will be mapped
		to the surface of this input. It must be the same geometry as input 2, this input is used for animating
		the generated points.

Note, if Generate is [Surface Area, Per Primitive] a point float attribute named Surface will be setted with values
[primNumber, r1, r2] where r1, r2 are uniform random numbers from 0 to 1.
*/

// Check methods [getNumInfoCHOPChans, getInfoCHOPChan, getInfoDATSize, getInfoDATEntries]
// if you want to output values to the Info CHOP/DAT

// To get more help about these functions, look at SOP_CPlusPlusBase.h
class SprinkleSOP : public SOP_CPlusPlusBase
{
public:
	SprinkleSOP(const OP_NodeInfo* info);
	virtual ~SprinkleSOP();

	virtual void		getGeneralInfo(SOP_GeneralInfo*, const OP_Inputs*, void*) override;

	virtual void		execute(SOP_Output*, const OP_Inputs*, void*) override;

	virtual void		executeVBO(SOP_VBOOutput*, const OP_Inputs*, void*) override;

	virtual void		setupParameters(OP_ParameterManager*, void*) override;

	virtual void		getErrorString(OP_String*, void*) override;

private:
	void				executeAreaScatter(const OP_SOPInput*);

	void				executePrimScatter(const OP_SOPInput*);

	void				executeVolumeScatter(const OP_SOPInput*);

	void				executeBoundingBoxScatter(const OP_SOPInput*);

	std::vector<Position>	mapSurfaceToSOP(const OP_SOPInput*);

	bool				addPointToPrim(const Position*, const SOP_PrimitiveInfo* prims, size_t primN);

	bool				addPointToBoundingBox(BoundingBox&);

	Position			getPointInBoundingBox(BoundingBox&);

	bool				addPointToVolume(const OP_SOPInput*, BoundingBox&);

	Parameters*		myParms;
	int64_t			myInputCook;
	std::mt19937	myRNG;
	std::string		myError;

	RandomPointsBuffer*	myPoints;
	VolSprinkleTree*	myVolSprinkleTree;
	std::vector<float>	mySurfaceAttribute;
};

#endif // !__SprinkleSOP__
