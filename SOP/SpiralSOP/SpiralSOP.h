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

#ifndef __SpiralSOP__
#define __SpiralSOP__

#include <vector>

#include "SOP_CPlusPlusBase.h"
#include "Parameters.h"

/*
This example implements a SOP outputs an Archimedean Spiral given the following
parameters:
	- Orientation:	One of [X Axis, Y Axis, Z Axis], which determines the orientation of the spiral.
	- Top Radius:	Radius at the top of the spiral.
	- Bottom Radius:	Radius at the bottom of the spiral.
	- Height:	How big is the spiral about the y axis.
	- Turns:	How many times the spiral spins.
	- Divisions:	The number of line segments that make up the spiral.
	- Output Geometry:	One of [Points, Line, Triangle Strip] which determines
					what kind of geometry we output.
	- Strip Width:	If Output Geometry is Triangle Stirp, this determines the width
				of the triangle strip.
	- GPU Direct:	If On, load geometry directly to GPU.

This SOP is a generator and it takes no input.
*/

// Check methods [getNumInfoCHOPChans, getInfoCHOPChan, getInfoDATSize, getInfoDATEntries]
// if you want to output values to the Info CHOP/DAT

// To get more help about these functions, look at SOP_CPlusPlusBase.h
class SpiralSOP : public TD::SOP_CPlusPlusBase
{
public:
	SpiralSOP(const TD::OP_NodeInfo* info);
	virtual ~SpiralSOP();

	virtual void		getGeneralInfo(TD::SOP_GeneralInfo*, const TD::OP_Inputs*, void*) override;

	virtual void		execute(TD::SOP_Output*, const TD::OP_Inputs*, void*) override;

	virtual void		executeVBO(TD::SOP_VBOOutput*, const TD::OP_Inputs*, void*) override;

	virtual void		setupParameters(TD::OP_ParameterManager*, void*) override;

private:
	void		handleParameters(const TD::OP_Inputs*);

	void		calculateOutputPoints();

	void		calculateSpiralPoints(std::vector<TD::Position>::iterator&, int numPts);

	void		calculateNormals(std::vector<TD::Vector>::iterator&, std::vector<TD::Position>::const_iterator&, int numPts);

	void		calculateTriangleStrip();

	std::vector<TD::Position>	myPointPos;
	std::vector<TD::Vector>		myNormals;
	std::vector<int32_t>	myLineStrip;
	std::vector<TD::TexCoord>	myLineStripTexture;
	TD::BoundingBox				myBoundingBox;

	//// Parameters
	OrientationMenuItems		myOrientation;
	double						myTopRad;
	double						myBotRad;
	double						myHeight;
	double						myTurns;
	int							myNumPoints;
	OutputgeometryMenuItems		myOutput;
	double						myStripWidth;
	
	Parameters myParms;
};

#endif // !__SpiralSOP__
