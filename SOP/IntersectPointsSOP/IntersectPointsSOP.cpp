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

#include "IntersectPointsSOP.h"
#include "Parameters.h"

#include <cassert>
#include <vector>
#include <array>

using namespace TD;

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
	customInfo.opType->setString("Intersectpoints");
	// English readable name
	customInfo.opLabel->setString("Intersect Points");
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
	return new IntersectPointsSOP(info);
}

DLLEXPORT
void
DestroySOPInstance(SOP_CPlusPlusBase* instance)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the CHOP using that instance is deleted, or
	// if the CHOP loads a different DLL
	delete (IntersectPointsSOP*)instance;
}

};

IntersectPointsSOP::IntersectPointsSOP(const OP_NodeInfo*)
{
};

IntersectPointsSOP::~IntersectPointsSOP()
{
};

void
IntersectPointsSOP::getGeneralInfo(SOP_GeneralInfo* ginfo, const TD::OP_Inputs* inputs, void*)
{
	// This will cause the node to cook every frame if the output is used
	ginfo->cookEveryFrameIfAsked = true;

	// Direct shape to GPU loading if asked 
	ginfo->directToGPU = false;
}

void
IntersectPointsSOP::execute(SOP_Output* output, const TD::OP_Inputs* inputs, void*)
{
	const OP_SOPInput* sop0 = inputs->getInputSOP(0);
	const OP_SOPInput* sop1 = inputs->getInputSOP(1);
	if (!sop0 || !sop1)
		return;

	copyPoints(output, sop0);
	copyAttributes(output, sop0);
	copyPrimitives(output, sop0);

	const Position* pos = sop0->getPointPositions();

	Color inside = myParms.evalInsidecolor(inputs);
	Color outside = myParms.evalOutsidecolor(inputs);

	std::vector<int> insideAttrib;
	insideAttrib.reserve(sop0->getNumPoints());
	for (int i = 0; i < sop0->getNumPoints(); ++i)
	{
		// OP_SOPInput* cp = sop1;
		insideAttrib.push_back(((OP_SOPInput*)sop1)->isInside(pos[i]));
		if (insideAttrib.at(i))
			output->setColor(inside, i);
		else
			output->setColor(outside, i);
	}

	SOP_CustomAttribData attrib{ "Inside", 1, AttribType::Int };
	attrib.intData = insideAttrib.data();
	output->setCustomAttribute(&attrib, sop0->getNumPoints());
}

void
IntersectPointsSOP::executeVBO(SOP_VBOOutput*, const TD::OP_Inputs*, void*)
{
}

void
IntersectPointsSOP::setupParameters(TD::OP_ParameterManager* manager, void*)
{
	myParms.setup(manager);
}

void
IntersectPointsSOP::getWarningString(OP_String* warning, void*)
{
	warning->setString(myWarningString.c_str());
	// Reset string after reporting it.
	myWarningString = "";
}

void 
IntersectPointsSOP::copyPoints(SOP_Output* out, const OP_SOPInput* in) const
{
	out->addPoints(in->getPointPositions(), in->getNumPoints());
}

void
IntersectPointsSOP::copyNormals(SOP_Output* out, const OP_SOPInput* in) const
{
	if (!in->hasNormals())
		return;

	const Vector*	normals = in->getNormals()->normals;
	int				numPts = out->getNumPoints();

	out->setNormals(normals, numPts, 0);
}

void 
IntersectPointsSOP::copyColors(SOP_Output* out, const OP_SOPInput* in) const
{
	if (!in->hasColors())
		return;

	const Color*	colors = in->getColors()->colors;
	int				numPts = out->getNumPoints();

	out->setColors(colors, numPts, 0);
}

void
IntersectPointsSOP::copyTextures(SOP_Output* out, const OP_SOPInput* in) const
{
	const TexCoord*	textures = in->getTextures()->textures;
	int				numLayers = in->getTextures()->numTextureLayers;
	int				numPts = out->getNumPoints();

	out->setTexCoords(textures, numPts, numLayers, 0);
}

void
IntersectPointsSOP::copyCustomAttributes(SOP_Output* out, const OP_SOPInput* in) const
{
	int		numPts = out->getNumPoints();
	for (int i = 0; i < in->getNumCustomAttributes(); ++i)
	{
		const SOP_CustomAttribData* customAttrib = in->getCustomAttribute(i);
		out->setCustomAttribute(customAttrib, numPts);
	}
}

void 
IntersectPointsSOP::copyAttributes(SOP_Output* output, const OP_SOPInput* sop) const
{
	copyNormals(output, sop);
	//copyColors(output, sop); // We do not need to copy colors for this SOP
	copyTextures(output, sop);
	copyCustomAttributes(output, sop);
}

void
IntersectPointsSOP::copyPrimitives(SOP_Output* out, const OP_SOPInput* in)
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