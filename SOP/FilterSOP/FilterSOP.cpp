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

#include "FilterSOP.h"

#include <cassert>

// Names of the parameters
constexpr static char CHOP_NAME[]	= "Translatechop";

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
	customInfo.opType->setString("Filter");
	// English readable name
	customInfo.opLabel->setString("Filter");
	// Information of the author of the node
	customInfo.authorName->setString("Gabriel Robels");
	customInfo.authorEmail->setString("support@derivative.ca");

	// This CHOP takes one input
	customInfo.minInputs = 1;
	customInfo.maxInputs = 1;
}

DLLEXPORT
SOP_CPlusPlusBase*
CreateSOPInstance(const OP_NodeInfo* info)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per CHOP that is using the .dll
	return new FilterSOP(info);
}

DLLEXPORT
void
DestroySOPInstance(SOP_CPlusPlusBase* instance)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the CHOP using that instance is deleted, or
	// if the CHOP loads a different DLL
	delete (FilterSOP*)instance;
}

};


FilterSOP::FilterSOP(const OP_NodeInfo*)
{
};

FilterSOP::~FilterSOP()
{
};

void
FilterSOP::getGeneralInfo(SOP_GeneralInfo* ginfo, const OP_Inputs* inputs, void*)
{
	// This will cause the node to cook every frame if the output is used
	// We set it to true otherwise the sop does not update when input sop changes
	ginfo->cookEveryFrameIfAsked = true;

	// Since it is a filter we cannot load geometry directly to GPU 
	ginfo->directToGPU = false;
}

void
FilterSOP::execute(SOP_Output* output, const OP_Inputs* inputs, void*)
{
	const OP_SOPInput*	sop = inputs->getInputSOP(0);
	if (!sop)
		return;

	Vector t{};
	const OP_CHOPInput* chop = inputs->getParCHOP(CHOP_NAME);
	if (!chop)
		myWarningString = "Translate CHOP not set.";
	else
		t = getTranslate(chop);

	copyPointsTranslated(output, sop, t);
	copyAttributes(output, sop);
	copyPrimitives(output, sop);
}

void
FilterSOP::executeVBO(SOP_VBOOutput* output, const OP_Inputs* inputs, void*)
{
	// Not Called since ginfo->directToGPU is false
}

void
FilterSOP::setupParameters(OP_ParameterManager* manager, void*)
{
	{
		OP_StringParameter	sp;

		sp.name = CHOP_NAME;
		sp.label = "Translate CHOP";
		sp.page = "Filter";

		OP_ParAppendResult res = manager->appendCHOP(sp);
		assert(res == OP_ParAppendResult::Success);
	}
}

void 
FilterSOP::getWarningString(OP_String* warning, void*)
{
	warning->setString(myWarningString.c_str());
	// Reset string after reporting it.
	myWarningString = "";
}

void 
FilterSOP::copyPointsTranslated(SOP_Output* output, const OP_SOPInput* sop, const Vector& t) const
{
	const Position*	inPos = sop->getPointPositions();
	for (int i = 0; i < sop->getNumPoints(); ++i)
	{
		output->addPoint(inPos[i] + t);
	}
}

void 
FilterSOP::copyAttributes(SOP_Output* output, const OP_SOPInput* sop) const
{
	copyNormals(output, sop);
	copyColors(output, sop);
	copyTextures(output, sop);
	copyCustomAttributes(output, sop);
}

void
FilterSOP::copyNormals(SOP_Output* out, const OP_SOPInput* in) const
{
	if (!in->hasNormals())
		return;

	const Vector*	normals = in->getNormals()->normals;
	int				numPts = out->getNumPoints();

	out->setNormals(normals, numPts, 0);
}

void 
FilterSOP::copyColors(SOP_Output* out, const OP_SOPInput* in) const
{
	if (!in->hasColors())
		return;

	const Color*	colors = in->getColors()->colors;
	int				numPts = out->getNumPoints();

	out->setColors(colors, numPts, 0);
}

void 
FilterSOP::copyTextures(SOP_Output* out, const OP_SOPInput* in) const
{
	const TexCoord*	textures = in->getTextures()->textures;
	int				numLayers = in->getTextures()->numTextureLayers;
	int				numPts = out->getNumPoints();

	out->setTexCoords(textures, numPts, numLayers, 0);
}

void 
FilterSOP::copyCustomAttributes(SOP_Output* out, const OP_SOPInput* in) const
{
	int		numPts = out->getNumPoints();
	for (int i = 0; i < in->getNumCustomAttributes(); ++i)
	{
		const SOP_CustomAttribData* customAttrib = in->getCustomAttribute(i);
		out->setCustomAttribute(customAttrib, numPts);
	}
}

void 
FilterSOP::copyPrimitives(SOP_Output* out, const OP_SOPInput* in)
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

Vector 
FilterSOP::getTranslate(const OP_CHOPInput* chop)
{
	Vector ret{};

	int		lastSample = chop->numSamples - 1;
	int		numChan = chop->numChannels;

	if (numChan > 2)
	{
		ret.z = chop->getChannelData(2)[lastSample];
	}
	else
	{
		myWarningString = "Translate CHOP should have at least 3 channels.";
	}

	if (numChan > 1)
		ret.y = chop->getChannelData(1)[lastSample];
	if (numChan > 0)
		ret.x = chop->getChannelData(0)[lastSample];

	return ret;
}
