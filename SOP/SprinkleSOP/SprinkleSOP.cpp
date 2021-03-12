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

#include "SprinkleSOP.h"
#include "Parameters.h"
#include "RandomPointsBuffer.h"
#include "VolSprinkleTree.h"

#include <cassert>
#include <cmath>
#include <random>
#include <algorithm>

namespace
{
	Vector
	toVector(const Position& p)
	{
		return Vector{ p.x, p.y, p.z };
	}

	bool 
	checkPrimitives(const OP_SOPInput* sop)
	{
		const SOP_PrimitiveInfo* prims = sop->myPrimsInfo;
		for (int i = 0; i < sop->getNumPrimitives(); ++i)
		{
			int nVert = prims[i].numVertices;
			if (prims[i].type != PrimitiveType::Polygon || nVert != 3)
				return false;
		}

		return true;
	}

	double
	distance(const Position& A, const Position& B)
	{
		Vector AB{ B.x - A.x, B.y - A.y, B.z - A.z };
		return AB.length();
	}

	double 
	calcArea(const Position& A, const Position& B, const Position& C)
	{
		Vector AB{ B.x - A.x, B.y - A.y, B.z - A.z };
		Vector AC{ C.x - A.x, C.y - A.y, C.z - A.z };
		const double ABnorm = AB.length();
		const double ACnorm = AC.length();
		double theta = std::acos(AB.dot(AC) / (ABnorm * ACnorm));
		return 0.5 * ABnorm * ACnorm * std::sin(theta);
	}

	Position 
	randPoint(const Position& pA, const Position& pB, const Position& pC, float r1, float r2)
	{
		Vector A = toVector(pA);
		Vector B = toVector(pB);
		Vector C = toVector(pC);
		float t = std::sqrt(r1);
		Vector P = A * (1 - t) + B * (t * (1 - r2)) + C * r2 * t;
		return Position{ P.x, P.y, P.z };

	}

	std::vector<double> 
	calcAreaVector(const SOP_PrimitiveInfo* prims, int numPrims, const Position* pos)
	{
		std::vector<double> ret(numPrims);
		double totalArea = 0.0;
		for (int i = 0; i < numPrims; ++i)
		{
			const int32_t* idx = prims[i].pointIndices;
			totalArea += calcArea(pos[idx[0]], pos[idx[1]], pos[idx[2]]);
			ret.at(i) = totalArea;
		}

		return ret;
	}

	BoundingBox 
	getBoundingBox(const Position* pos, int numPoints)
	{
		if (numPoints == 0)
			return BoundingBox(Position(), Position());

		Position min, max;
		min = max = pos[0];

		for (int i = 1; i < numPoints; ++i)
		{
			const Position& p = pos[i];
			if (p.x > max.x)
				max.x = p.x;
			if (p.y > max.y)
				max.y = p.y;
			if (p.z > max.z)
				max.z = p.z;
			if (p.x < min.x)
				min.x = p.x;
			if (p.y < min.y)
				min.y = p.y;
			if (p.z < min.z)
				min.z = p.z;
		}

		return BoundingBox(min, max);
	}
}

// These functions are basic C function, which the DLL loader can find
// much easier than finding a C++ Class.
// The DLLEXPORT prefix is needed so the compile exports these functions from the .dll
// you are creating
extern "C"
{

DLLEXPORT
void
FillSOPPluginInfo(SOP_PluginInfo *info)
{
	// For more information on CHOP_PluginInfo see CHOP_CPlusPlusBase.h

	// Always set this to CHOPCPlusPlusAPIVersion.
	info->apiVersion = SOPCPlusPlusAPIVersion;

	// For more information on OP_CustomOPInfo see CPlusPlus_Common.h
	OP_CustomOPInfo& customInfo = info->customOPInfo;

	// Unique name of the node which starts with an upper case letter, followed by lower case letters or numbers
	customInfo.opType->setString("Sprinkle");
	// English readable name
	customInfo.opLabel->setString("Sprinkle");
	// Information of the author of the node
	customInfo.authorName->setString("Gabriel Robels");
	customInfo.authorEmail->setString("support@derivative.ca");

	// This SOP takes exactly one input
	customInfo.minInputs = 1;
	customInfo.maxInputs = 2;
}

DLLEXPORT
SOP_CPlusPlusBase*
CreateSOPInstance(const OP_NodeInfo* info)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per CHOP that is using the .dll
	return new SprinkleSOP(info);
}

DLLEXPORT
void
DestroySOPInstance(SOP_CPlusPlusBase* instance)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the CHOP using that instance is deleted, or
	// if the CHOP loads a different DLL
	delete (SprinkleSOP*)instance;
}

};

SprinkleSOP::SprinkleSOP(const OP_NodeInfo*) : 
	myRNG{}, myPoints{ nullptr }, myVolSprinkleTree{ nullptr }, myInputCook {}
{
};

SprinkleSOP::~SprinkleSOP()
{
	delete myPoints;
	delete myVolSprinkleTree;
};

void
SprinkleSOP::getGeneralInfo(SOP_GeneralInfo* ginfo, const OP_Inputs* inputs, void*)
{
	// This will cause the node to cook every frame if the output is used
	ginfo->cookEveryFrameIfAsked = true;
	// Direct shape to GPU loading if asked 
	ginfo->directToGPU = false;
}

void
SprinkleSOP::execute(SOP_Output* output, const OP_Inputs* inputs, void*)
{
	const OP_SOPInput* sop = inputs->getInputSOP(0);
	if (!sop)
		return;

	myParms.evalParms(inputs);
	unsigned int* seedInt = reinterpret_cast<unsigned int*>(&myParms.seed);
	myRNG.seed(*seedInt);

	if (!checkPrimitives(sop))
	{
		myError = "Input geometry is not a triangulated polygon.";
		return;
	}

	// Debug code
	if (false) 
	{
		const Position* p = sop->getPointPositions();
		BoundingBox& bb = getBoundingBox(p, sop->getNumPoints());
		if (myInputCook != sop->totalCooks)
		{
			delete myVolSprinkleTree;
			myVolSprinkleTree = new VolSprinkleTree(sop, bb);
			myInputCook = sop->totalCooks;
		}
		myVolSprinkleTree->outputTest(output);
		return;
	}

	if (!myPoints || myParms.changed || myInputCook != sop->totalCooks)
	{
		delete myPoints;
		mySurfaceAttribute.clear();
		myPoints = new RandomPointsBuffer(myParms.forcedistance, myParms.pointdistance);

		switch (myParms.generate)
		{
		case Generate::Density:
			executeAreaScatter(sop);
			break;
		case Generate::Primitive:
			executePrimScatter(sop);
			break;
		case Generate::BoundingBox:
			executeBoundingBoxScatter(sop);
			break;
		case Generate::Volume:
			executeVolumeScatter(sop);
			break;
		}

		myInputCook = sop->totalCooks;
	}

	bool surfaceGen = myParms.generate == Generate::Density || myParms.generate == Generate::Primitive;

	const OP_SOPInput* sop1 = inputs->getInputSOP(1);
	if (surfaceGen && sop1)
	{
		std::vector<Position>&& vec = mapSurfaceToSOP(sop1);
		output->addPoints(vec.data(), static_cast<int32_t>(vec.size()));
	}
	else
	{
		std::vector<Position>& vec = myPoints->getVector();
		output->addPoints(vec.data(), static_cast<int32_t>(vec.size()));
	}

	if (surfaceGen)
	{
		SOP_CustomAttribData data("Surface", 3, AttribType::Float);
		data.floatData = mySurfaceAttribute.data();
		output->setCustomAttribute(&data, output->getNumPoints());
	}
}

void
SprinkleSOP::executeVBO(SOP_VBOOutput*, const OP_Inputs*, void*)
{
}

void
SprinkleSOP::setupParameters(OP_ParameterManager* manager, void*)
{
	myParms.setupParms(manager);
}

void
SprinkleSOP::getErrorString(OP_String* error, void*)
{
	error->setString(myError.c_str());
	// Reset string after reporting it.
	myError = "";
}

void
SprinkleSOP::executeAreaScatter(const OP_SOPInput* sop)
{
	const SOP_PrimitiveInfo* prims = sop->myPrimsInfo;
	const Position* pos = sop->getPointPositions();

	std::vector<double> cumulativeA = calcAreaVector(prims, sop->getNumPrimitives(), pos);
	double totalA = cumulativeA.back();

	std::uniform_real_distribution<> uni(0.0, 1.0);
	for (int i = 0; i < myParms.pointcount; ++i)
	{
		bool success = false;
		for (int tries = 0; tries < myParms.pointcount * 0.05; ++tries)
		{
			size_t prim = std::upper_bound(cumulativeA.begin(), cumulativeA.end(), uni(myRNG) * totalA) - cumulativeA.begin();
			if (addPointToPrim(pos, prims, prim))
			{
				success = true;
				break;
			}
		}
		if (!success)
			return;

	}
}

void 
SprinkleSOP::executePrimScatter(const OP_SOPInput* sop)
{
	const SOP_PrimitiveInfo* prims = sop->myPrimsInfo;
	const Position* pos = sop->getPointPositions();

	float pointsPerPrim = myParms.pointcount / static_cast<float>(sop->getNumPrimitives());
	int pointsGen = 0;

	for (int i = 0; i < sop->getNumPrimitives(); ++i)
	{
		for (; pointsGen < (i+1) * pointsPerPrim; ++pointsGen)
		{
			if (!addPointToPrim(pos, prims, i))
				break;
		}
	}

}

void 
SprinkleSOP::executeVolumeScatter(const OP_SOPInput* sop)
{
	const SOP_PrimitiveInfo* prims = sop->myPrimsInfo;
	const Position* pos = sop->getPointPositions();
	BoundingBox bBox = getBoundingBox(pos, sop->getNumPoints());

	if (!myVolSprinkleTree || myInputCook != sop->totalCooks)
	{
		delete myVolSprinkleTree;
		myVolSprinkleTree = new VolSprinkleTree(sop, bBox);
	}

	for (int i = 0; i < myParms.pointcount; ++i)
	{
		bool success = false;
		for (int tries = 0; tries < myParms.pointcount * 0.1 + 3; ++tries)
		{
			if (addPointToVolume(sop, bBox))
			{
				success = true;
				break;
			}
		}
		if (!success)
			break;
	}
}

void 
SprinkleSOP::executeBoundingBoxScatter(const OP_SOPInput* sop)
{
	const Position* pos = sop->getPointPositions();
	BoundingBox bBox = getBoundingBox(pos, sop->getNumPoints());
	for (int i = 0; i < myParms.pointcount; ++i)
	{
		bool success = false;
		for (int tries = 0; tries < myParms.pointcount * 0.05; ++tries)
		{
			if (addPointToBoundingBox(bBox))
			{
				success = true;
				break;
			}
		}
		if (!success)
			return;
	}
}

std::vector<Position>
SprinkleSOP::mapSurfaceToSOP(const OP_SOPInput* sop)
{
	std::vector<Position> ret{};
	const Position* pos = sop->getPointPositions();
	const SOP_PrimitiveInfo* prims = sop->myPrimsInfo;

	for (int i = 0; i < mySurfaceAttribute.size(); i += 3)
	{
		int primN = static_cast<int>(mySurfaceAttribute.at(i));
		const int32_t* idx = prims[primN].pointIndices;
		float r[2] = { mySurfaceAttribute.at(i + 1), mySurfaceAttribute.at(i + 2) };
		ret.emplace_back(randPoint(pos[idx[0]], pos[idx[1]], pos[idx[2]], r[0], r[1]));
	}
	return ret;
}

bool
SprinkleSOP::addPointToPrim(const Position* pos, const SOP_PrimitiveInfo* prims, size_t primN)
{
	const SOP_PrimitiveInfo& prim = prims[primN];
	const int32_t* idx = prim.pointIndices;
	std::uniform_real_distribution<> uni(0.0, 1.0);

	float r[2] = { static_cast<float>(uni(myRNG)), static_cast<float>(uni(myRNG)) };

	Position newP = randPoint(pos[idx[0]], pos[idx[1]], pos[idx[2]], r[0], r[1]);
	
	if (myPoints->addIfPointFits(newP))
	{
		mySurfaceAttribute.push_back(static_cast<float>(primN));
		mySurfaceAttribute.push_back(r[0]);
		mySurfaceAttribute.push_back(r[1]);
		return true;
	}
	return false;
}

bool 
SprinkleSOP::addPointToBoundingBox(BoundingBox& bb)
{
	Position p = getPointInBoundingBox(bb);
	return myPoints->addIfPointFits(p);
}

Position 
SprinkleSOP::getPointInBoundingBox(BoundingBox& bb)
{
	Position p;
	std::uniform_real_distribution<> uni(0.0, 1.0);

	p.x = static_cast<float>(uni(myRNG) * bb.sizeX() + bb.minX);
	p.y = static_cast<float>(uni(myRNG) * bb.sizeY() + bb.minY);
	p.z = static_cast<float>(uni(myRNG) * bb.sizeZ() + bb.minZ);
	return p;
}

bool 
SprinkleSOP::addPointToVolume(const OP_SOPInput* sop, BoundingBox& bb)
{
	std::uniform_real_distribution<> uni(0.0, 1.0);
	auto newRNG = [&]() {
		return static_cast<float>(uni(myRNG));
	};
	//Position p = getPointInBoundingBox(bb);
	Position p = myVolSprinkleTree->getPoint(newRNG(), newRNG(), newRNG(), newRNG());
	
	if (sop->isInside(p))
	{
		return myPoints->addIfPointFits(p);
	}

	return false;
}
