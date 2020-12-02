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

#include "SpiralSOP.h"

#include <cassert>
#include <cmath>
#include <numeric>

static constexpr double PI = 3.141592653589793238463;

// Names of the parameters
constexpr static char	ORIENTATION_NAME[] = "Orientation";
constexpr static char	TOPRADIUS_NAME[] = "Topradius";
constexpr static char	BOTRADIUS_NAME[] = "Bottomradius";
constexpr static char	HEIGHT_NAME[] = "Height";
constexpr static char	TURNS_NAME[] = "Turns";
constexpr static char	POINTS_NAME[] = "Divisions";
constexpr static char	OUTPUT_NAME[] = "Output";
constexpr static char	STRIPWIDTH_NAME[] = "Stripwidth";
constexpr static char	GPUDIRECT_NAME[] = "Gpudirect";

enum class
OutGeometry
{
	Points = 0,
	Line = 1,
	TriangleStrip = 2
};

enum class
Orientation
{
	X,
	Y,
	Z
};

namespace
{
	Vector PerpBofA(const Vector& A, const Vector& B)
	{
		Vector ret = B;
		ret -= A * (A.dot(B) / A.length() / A.length());
		return ret;

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
	customInfo.opType->setString("Spiral");
	// English readable name
	customInfo.opLabel->setString("Spiral");
	// Information of the author of the node
	customInfo.authorName->setString("Gabriel Robels");
	customInfo.authorEmail->setString("support@derivative.ca");

	// This SOP takes no input
	customInfo.minInputs = 0;
	customInfo.maxInputs = 0;
}

DLLEXPORT
SOP_CPlusPlusBase*
CreateSOPInstance(const OP_NodeInfo* info)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per CHOP that is using the .dll
	return new SpiralSOP(info);
}

DLLEXPORT
void
DestroySOPInstance(SOP_CPlusPlusBase* instance)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the CHOP using that instance is deleted, or
	// if the CHOP loads a different DLL
	delete (SpiralSOP*)instance;
}

};

SpiralSOP::SpiralSOP(const OP_NodeInfo*) :
	myBoundingBox{Position(), Position()}
{
};

SpiralSOP::~SpiralSOP()
{
};

void
SpiralSOP::getGeneralInfo(SOP_GeneralInfo* ginfo, const OP_Inputs* inputs, void*)
{
	// This will cause the node to cook every frame if the output is used
	ginfo->cookEveryFrameIfAsked = false;

	// Direct shape to GPU loading if asked 
	ginfo->directToGPU = inputs->getParInt(GPUDIRECT_NAME) ? true : false;
}

void
SpiralSOP::execute(SOP_Output* output, const OP_Inputs* inputs, void*)
{
	handleParameters(inputs);

	calculateOutputPoints();
	output->addPoints(myPointPos.data(), myNumPoints);
	output->setNormals(myNormals.data(), myNumPoints, 0);

	switch (myOutput)
	{
		case OutGeometry::Points:
		{
			output->addParticleSystem(myNumPoints, 0);
			break;
		}
		case OutGeometry::Line:
		{
			std::vector<int32_t> line(myNumPoints);
			std::iota(line.begin(), line.end(), 0); // Fill a vector with sequencial indices starting at 0
			output->addLine(line.data(), myNumPoints);
			break;
		}
		case OutGeometry::TriangleStrip:
		{
			calculateTriangleStrip();
			if (myNumPoints > 2)
			{
				output->addTriangles(myLineStrip.data(), myNumPoints - 2);
				for (int i = 0; i < myLineStripTexture.size(); i += 1)
				{
					output->setTexCoord(&myLineStripTexture.at(i), 1, i);
				}
			}
			break;
		}
	}

	output->setBoundingBox(myBoundingBox);
}

void
SpiralSOP::executeVBO(SOP_VBOOutput* output, const OP_Inputs* inputs, void*)
{
	handleParameters(inputs);

	output->enableNormal();
	output->enableTexCoord(1);

	if (myOutput == OutGeometry::TriangleStrip)
	{
		output->allocVBO(myNumPoints * 2, myNumPoints * 6, VBOBufferMode::Static);
	}
	else
	{
		output->allocVBO(myNumPoints, myNumPoints, VBOBufferMode::Static);
	}

	calculateOutputPoints();
	memcpy(output->getPos(), myPointPos.data(), myNumPoints * sizeof(Position));
	Vector* outN = output->getNormals();
	for (int i = 0; i < myNumPoints; i += 1, outN += 1)
	{
		*outN = myNormals.at(i) * -1.f;
	}

	std::vector<int32_t> seqNum(myNumPoints);
	std::iota(seqNum.begin(), seqNum.end(), 0); // Fill a vector with sequencial indices starting at 0
	
	switch (myOutput)
	{
		case OutGeometry::Points:
		{
			memcpy(output->addParticleSystem(myNumPoints), seqNum.data(), myNumPoints * sizeof(int32_t));
			break;
		}
		case OutGeometry::Line:
		{
			memcpy(output->addLines(myNumPoints), seqNum.data(), myNumPoints * sizeof(int32_t));
			break;
		}
		case OutGeometry::TriangleStrip:
		{
			calculateTriangleStrip();
			memcpy(output->addTriangles(myNumPoints - 2), myLineStrip.data(), (myNumPoints - 2) * 3 * sizeof(int32_t));
			memcpy(output->getTexCoords(), myLineStripTexture.data(), myNumPoints * sizeof(TexCoord));
			break;
		}
	}

	output->setBoundingBox(myBoundingBox);
	output->updateComplete();
}

void
SpiralSOP::setupParameters(OP_ParameterManager* manager, void*)
{
	{
		OP_StringParameter sp;

		sp.name = ORIENTATION_NAME;
		sp.label = "Orientation";
		sp.page = "Spiral";
		sp.defaultValue = "X Axis";

		const char* names[] = { "X", "Y", "Z" };
		const char* labels[] = { "X Axis", "Y Axis", "Z Axis" };

		OP_ParAppendResult res = manager->appendMenu(sp, 3, names, labels);
		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter	np;

		np.name = TOPRADIUS_NAME;
		np.label = "Top Radius";
		np.page = "Spiral";

		np.defaultValues[0] = 0.3f;
		np.minSliders[0] = 0.0f;
		np.maxSliders[0] = 10.0f;

		OP_ParAppendResult res = manager->appendFloat(np);
		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter	np;

		np.name = BOTRADIUS_NAME;
		np.label = "Bottom Radius";
		np.page = "Spiral";

		np.defaultValues[0] = 1.0f;
		np.minSliders[0] = 0.0f;
		np.maxSliders[0] = 10.0f;

		OP_ParAppendResult res = manager->appendFloat(np);
		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter	np;

		np.name = HEIGHT_NAME;
		np.label = "Height";
		np.page = "Spiral";

		np.defaultValues[0] = 2.0f;
		np.minSliders[0] = 0.0f;
		np.maxSliders[0] = 20.0f;

		OP_ParAppendResult res = manager->appendFloat(np);
		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter	np;

		np.name = TURNS_NAME;
		np.label = "Turns";
		np.page = "Spiral";

		np.defaultValues[0] = 5.0f;
		np.minSliders[0] = 0.0f;
		np.maxSliders[0] = 10.0f;

		OP_ParAppendResult res = manager->appendFloat(np);
		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter	np;

		np.name = POINTS_NAME;
		np.label = "Divisions";
		np.page = "Spiral";

		np.defaultValues[0] = 100;
		np.minSliders[0] = 0;
		np.maxSliders[0] = 500;
		np.minValues[0] = 0;

		OP_ParAppendResult res = manager->appendInt(np);
		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter	sp;

		sp.name = OUTPUT_NAME;
		sp.label = "Output Geometry";
		sp.page = "Spiral";

		const char* names[] = { "Points", "Line", "Trianglestrip" };
		const char* labels[] = { "Points", "Line", "Triangle Strip" };

		sp.defaultValue = "Line";

		OP_ParAppendResult res = manager->appendMenu(sp, 3, names, labels);
		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter	np;

		np.name = STRIPWIDTH_NAME;
		np.label = "Strip Width";
		np.page = "Spiral";

		np.defaultValues[0] = 0.2f;
		np.minSliders[0] = 0;
		np.maxSliders[0] = 5.0f;

		OP_ParAppendResult res = manager->appendFloat(np);
		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter	np;

		np.name = GPUDIRECT_NAME;
		np.label = "GPU Direct";
		np.page = "Spiral";
		np.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(np);
		assert(res == OP_ParAppendResult::Success);
	}
}

void 
SpiralSOP::handleParameters(const OP_Inputs* in)
{
	myOrientation = static_cast<Orientation>(in->getParInt(ORIENTATION_NAME));
	myTopRad = in->getParDouble(TOPRADIUS_NAME);
	myBotRad = in->getParDouble(BOTRADIUS_NAME);
	myHeight = in->getParDouble(HEIGHT_NAME);
	myTurns = in->getParDouble(TURNS_NAME);
	myNumPoints = in->getParInt(POINTS_NAME);
	myOutput = static_cast<OutGeometry>(in->getParInt(OUTPUT_NAME));

	bool isTriangleStrip = myOutput == OutGeometry::TriangleStrip;
	in->enablePar(STRIPWIDTH_NAME, isTriangleStrip);
	myStripWidth = isTriangleStrip ? in->getParDouble(STRIPWIDTH_NAME) : 0.0f;
	myNumPoints *= isTriangleStrip ? 2 : 1;	// Triangle strip needs two spirals so multiply numPoints by 2

	// Calculate bounding box
	float maxRad = static_cast<float>(fabs(myBotRad) > fabs(myTopRad) ? fabs(myBotRad) : fabs(myTopRad));
	switch (myOrientation)
	{
		case Orientation::X:
			myBoundingBox = BoundingBox(static_cast<float>(-myHeight / 2), -maxRad, -maxRad, static_cast<float>(myHeight / 2), maxRad, maxRad);
			break;
		default:
		case Orientation::Y:
			myBoundingBox = BoundingBox(-maxRad, static_cast<float>(-myHeight / 2), -maxRad, maxRad, static_cast<float>(myHeight / 2), maxRad);
			break;
		case Orientation::Z:
			myBoundingBox = BoundingBox(-maxRad, -maxRad, static_cast<float>(-myHeight / 2), maxRad, maxRad, static_cast<float>(myHeight / 2));
			break;
	}

}

void
SpiralSOP::calculateOutputPoints()
{
	myPointPos.resize(myNumPoints);
	myNormals.resize(myNumPoints);

	if (myOutput == OutGeometry::TriangleStrip)
	{

		int halfPts = myNumPoints / 2;	// myNumPoints is divisible by 2 when triangle strip
		calculateSpiralPoints(myPointPos.begin(), halfPts);
		calculateNormals(myNormals.begin(), myPointPos.cbegin(), halfPts);

		myBotRad -= myStripWidth;
		myTopRad -= myStripWidth;
		calculateSpiralPoints(myPointPos.begin() + halfPts, halfPts);
		calculateNormals(myNormals.begin() + halfPts, myPointPos.cbegin() + halfPts, halfPts);
	}
	else
	{
		calculateSpiralPoints(myPointPos.begin(), myNumPoints);
		calculateNormals(myNormals.begin(), myPointPos.cbegin(), myNumPoints);
	}
}

void 
SpiralSOP::calculateSpiralPoints(std::vector<Position>::iterator& it, int numPts)
{
	const double deltaA = (2 * PI * myTurns) / numPts;
	const double deltaR = (myTopRad - myBotRad) / numPts;
	const double deltaH = myHeight / numPts;

	double	rad = myBotRad;

	for (int i = 0; i < numPts; ++i, ++it, rad += deltaR)
	{
		const float x = static_cast<float>(rad * cos(i * deltaA));
		const float y = static_cast<float>(rad * sin(i * deltaA));
		const float h = static_cast<float>(i * deltaH - myHeight / 2);
		switch (myOrientation)
		{
			case Orientation::X:
				*it = Position(h, x, y);
				break;
			default:
			case Orientation::Y:
				*it = Position(x, h, y);
				break;
			case Orientation::Z:
				*it = Position(x, y, h);
				break;
		}
	}
}

void
SpiralSOP::calculateNormals(std::vector<Vector>::iterator& N, std::vector<Position>::const_iterator& pts, int numPts)
{
	Vector reference;
	switch (myOrientation)
	{
	case Orientation::X:
		reference = Vector(-1, 0, 0);
		break;
	default:
	case Orientation::Y:
		reference = Vector(0, 1, 0);
		break;
	case Orientation::Z:
		reference = Vector(0, 0, -1);
		break;
	}

	Vector	direction{ pts->x - (pts + 1)->x , pts->y - (pts + 1)->y, pts->z - (pts + 1)->z };
	*N++ = PerpBofA(direction, reference);
	++pts;

	for (int i = 1; i < numPts; ++i, ++pts)
	{
		direction = Vector(pts->x - (pts - 1)->x, pts->y - (pts - 1)->y, pts->z - (pts - 1)->z);
		*N++ = PerpBofA(direction, reference);
	}
}

void 
SpiralSOP::calculateTriangleStrip()
{
	myLineStrip.resize(myNumPoints * 3);
	int32_t* data = myLineStrip.data();
	int halfPts = myNumPoints / 2;	// myNumPoints is divisible by 2 when triangle strip
	for (int i = 0; i < halfPts - 1; ++i, data += 6)
	{
		int32_t tmp[6] = { i, i + 1, halfPts + i, halfPts + i, i + 1, halfPts + i + 1 };
		memcpy(data, tmp, 6 * sizeof(int32_t));
	}

	if (myNumPoints != myLineStripTexture.size())
	{
		const float dV = 1.f / (halfPts - 1);
		float V = 0;
		myLineStripTexture.resize(myNumPoints);
		for (int i = 0; i < halfPts; i += 1, V += dV)
		{
			myLineStripTexture.at(i) = TexCoord(V, 0.f, 0.f);
			myLineStripTexture.at(i + halfPts) = TexCoord(V, 1.f, 0.f);
		}
	}
}
