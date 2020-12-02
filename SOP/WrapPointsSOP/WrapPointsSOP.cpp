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

#include "WrapPointsSOP.h"
#include "Parameters.h"

#include <cassert>

namespace
{
	Position 
	getAveragePosition(const Position* pts, int numPts)
	{
		Position avg{};
		for (int i = 0; i < numPts; ++i)
		{
			avg.x += pts[i].x;
			avg.y += pts[i].y;
			avg.z += pts[i].z;
		}
		avg.x /= numPts;
		avg.y /= numPts;
		avg.z /= numPts;
		
		return avg;
	}

	Vector
	getDirection(const Position& origin, const Position& destination)
	{
		return Vector{ 
			destination.x - origin.x, 
			destination.y - origin.y, 
			destination.z - origin.z };
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
	customInfo.opType->setString("Wrappoints");
	// English readable name
	customInfo.opLabel->setString("Wrap Points");
	// Information of the author of the node
	customInfo.authorName->setString("Gabriel Robels");
	customInfo.authorEmail->setString("support@derivative.ca");

	// This SOP takes exactly one input
	customInfo.minInputs = 2;
	customInfo.maxInputs = 2;
}

DLLEXPORT
SOP_CPlusPlusBase*
CreateSOPInstance(const OP_NodeInfo* info)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per CHOP that is using the .dll
	return new WrapPointsSOP(info);
}

DLLEXPORT
void
DestroySOPInstance(SOP_CPlusPlusBase* instance)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the CHOP using that instance is deleted, or
	// if the CHOP loads a different DLL
	delete (WrapPointsSOP*)instance;
}

};

WrapPointsSOP::WrapPointsSOP(const OP_NodeInfo*) : myParms{ new Parameters() }
{
};

WrapPointsSOP::~WrapPointsSOP()
{
	delete myParms;
};

void
WrapPointsSOP::getGeneralInfo(SOP_GeneralInfo* ginfo, const OP_Inputs* inputs, void*)
{
	// This will cause the node to cook every frame if the output is used
	ginfo->cookEveryFrameIfAsked = true;

	// Direct shape to GPU loading if asked 
	ginfo->directToGPU = false;
}

void
WrapPointsSOP::execute(SOP_Output* output, const OP_Inputs* inputs, void*)
{
	const OP_SOPInput* sop0 = inputs->getInputSOP(0);
	const OP_SOPInput* sop1 = inputs->getInputSOP(1);
	if (!sop0 || !sop1)
		return;

	myParms->evalParms(inputs);

	switch (myParms->rays)
	{
		default:
		case Rays::Parallel:
		{
			double* t = myParms->direction;
			Vector direction = { static_cast<float>(t[0]), static_cast<float>(t[1]), static_cast<float>(t[2]) };
			castParallel(output, sop0, sop1, direction);
			break;
		}
		case Rays::Radial:
		{
			double* t = myParms->origin;
			Position origin = { static_cast<float>(t[0]), static_cast<float>(t[1]), static_cast<float>(t[2]) };
			castRadial(output, sop0, sop1, origin);
			break;
		}
	}

	copyAttributes(output, sop0);
	copyPrimitives(output, sop0);
}

void
WrapPointsSOP::executeVBO(SOP_VBOOutput*, const OP_Inputs*, void*)
{
}

void
WrapPointsSOP::setupParameters(OP_ParameterManager* manager, void*)
{
	myParms->setupParms(manager);
}

void
WrapPointsSOP::getWarningString(OP_String* warning, void*)
{
	warning->setString(myWarningString.c_str());
	// Reset string after reporting it.
	myWarningString = "";
}

void 
WrapPointsSOP::castParallel(SOP_Output* output, const OP_SOPInput* sop0, const OP_SOPInput* sop1, Vector dir)
{
	const Position* pos = sop0->getPointPositions();
	const Vector* norm0 = sop0->getNormals()->normals;

	if (myParms->reverse)
		dir = dir * -1;

	for (int i = 0; i < sop0->getNumPoints(); ++i)
	{
		castPoint(output, pos, norm0, i, sop1, dir);
	}
}

void 
WrapPointsSOP::castRadial(SOP_Output* output, const OP_SOPInput* sop0, const OP_SOPInput* sop1, Position destination)
{
	const Position* pos = sop0->getPointPositions();
	const Vector* norm0 = sop0->getNormals()->normals;

	Vector direction;
	for (int i = 0; i < sop0->getNumPoints(); ++i)
	{
		if (myParms->reverse)
			direction = getDirection(destination, pos[i]);
		else
			direction = getDirection(pos[i], destination);
		castPoint(output, pos, norm0, i, sop1, direction);
	}
}

void 
WrapPointsSOP::copyAttributes(SOP_Output* output, const OP_SOPInput* sop) const
{
	copyColors(output, sop);
	copyTextures(output, sop);
	copyCustomAttributes(output, sop);
}

void 
WrapPointsSOP::copyColors(SOP_Output* out, const OP_SOPInput* in) const
{
	if (!in->hasColors())
		return;

	const Color*	colors = in->getColors()->colors;
	int				numPts = out->getNumPoints();

	out->setColors(colors, numPts, 0);
}

void
WrapPointsSOP::copyTextures(SOP_Output* out, const OP_SOPInput* in) const
{
	const TexCoord*	textures = in->getTextures()->textures;
	int				numLayers = in->getTextures()->numTextureLayers;
	int				numPts = out->getNumPoints();

	out->setTexCoords(textures, numPts, numLayers, 0);
}

void
WrapPointsSOP::copyCustomAttributes(SOP_Output* out, const OP_SOPInput* in) const
{
	int		numPts = out->getNumPoints();
	for (int i = 0; i < in->getNumCustomAttributes(); ++i)
	{
		const SOP_CustomAttribData* customAttrib = in->getCustomAttribute(i);
		out->setCustomAttribute(customAttrib, numPts);
	}
}

void
WrapPointsSOP::copyPrimitives(SOP_Output* out, const OP_SOPInput* in)
{
	const SOP_PrimitiveInfo* prims = in->myPrimsInfo;
	bool isTriangulated = true;

	for (int i = 0; i < in->getNumPrimitives(); ++i)
	{
		int				nVertices = prims[i].numVertices;
		const int32_t*	indices = prims[i].pointIndices;

		if (prims[i].type != PrimitiveType::Polygon || nVertices > 3)
		{
			isTriangulated = false;
		}

		switch (nVertices)
		{

		case 1:
		{
			out->addParticleSystem(1, indices[0]);
			break;
		}
		case 2:
		{
			out->addLine(indices, 2);
			break;
		}
		case 3:
		{
			out->addTriangles(indices, 1);
		}
		default:
		{
			int32_t* tmp = new int32_t[nVertices + 1];
			memcpy(tmp, indices, nVertices * sizeof(int32_t));
			tmp[nVertices] = indices[0];

			out->addLine(tmp, nVertices + 1);

			delete[] tmp;
		}
		}
	}

	if (!isTriangulated)
	{
		myWarningString = "Input geometry is not a triangulated polygon.";
	}
}

void
WrapPointsSOP::castPoint(SOP_Output* output, const Position* pos, const Vector* normals, int index, const OP_SOPInput* geo, Vector dir)
{
	Position hitP{};
	float hitLength{}, hitU{}, hitV{};
	int hitPrimitiveIndex{};
	Vector hitNormal{};
	Position pt = pos[index];

	if (geo->sendRay(pt, dir, hitP, hitLength, hitNormal, hitU, hitV, hitPrimitiveIndex))
	{
		output->addPoint(pt + dir * static_cast<float>(hitLength * myParms->scale));
		output->setNormal(hitNormal, index);
		output->setColor(myParms->hitcolor, index);
	}
	else
	{
		output->addPoint(pt);
		output->setNormal(normals[index], index);
		output->setColor(myParms->misscolor, index);
	}
}