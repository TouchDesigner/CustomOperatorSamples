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

#include "TimeSliceGeneratorCHOP.h"

#include <cassert>

static constexpr double PI = 3.141592653589793238463;

// Names of the parameters
constexpr static char SPEED_NAME[]		= "Frequency";
constexpr static char APPLYSCALE_NAME[]	= "Applyscale";
constexpr static char SCALE_NAME[]		= "Scale";
constexpr static char SHAPE_NAME[]		= "Type";

enum class
Shape
{
	Sine = 0,
	Square = 1,
	Ramp = 2
};

// These functions are basic C function, which the DLL loader can find
// much easier than finding a C++ Class.
// The DLLEXPORT prefix is needed so the compile exports these functions from the .dll
// you are creating
extern "C"
{

DLLEXPORT
void
FillCHOPPluginInfo(CHOP_PluginInfo *info)
{
	// For more information on CHOP_PluginInfo see CHOP_CPlusPlusBase.h

	// Always set this to CHOPCPlusPlusAPIVersion.
	info->apiVersion = CHOPCPlusPlusAPIVersion;

	// For more information on OP_CustomOPInfo see CPlusPlus_Common.h
	OP_CustomOPInfo& customInfo = info->customOPInfo;

	// Unique name of the node which starts with an upper case letter, followed by lower case letters or numbers
	customInfo.opType->setString("Timeslicegenerator");
	// English readable name
	customInfo.opLabel->setString("Time Slice Generator");
	// Information of the author of the node
	customInfo.authorName->setString("Gabriel Robels");
	customInfo.authorEmail->setString("support@derivative.ca");

	// This CHOP takes no inputs
	customInfo.minInputs = 0;
	customInfo.maxInputs = 0;
}

DLLEXPORT
CHOP_CPlusPlusBase*
CreateCHOPInstance(const OP_NodeInfo* info)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per CHOP that is using the .dll
	return new TimeSliceGeneratorCHOP(info);
}

DLLEXPORT
void
DestroyCHOPInstance(CHOP_CPlusPlusBase* instance)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the CHOP using that instance is deleted, or
	// if the CHOP loads a different DLL
	delete (TimeSliceGeneratorCHOP*)instance;
}

};


TimeSliceGeneratorCHOP::TimeSliceGeneratorCHOP(const OP_NodeInfo*) : myOffset()
{

}

TimeSliceGeneratorCHOP::~TimeSliceGeneratorCHOP()
{

}

void
TimeSliceGeneratorCHOP::getGeneralInfo(CHOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	// This will cause the node to cook every frame if the output is used
	ginfo->cookEveryFrameIfAsked = true;

	// Note: To disable timeslicing you'll need to turn this off, as well as ensure that
	// getOutputInfo() returns true, and likely also set the info->numSamples to how many
	// samples you want to generate for this CHOP. Otherwise it'll take on length of the
	// input CHOP, which may be timesliced.
	ginfo->timeslice = true;
}

bool
TimeSliceGeneratorCHOP::getOutputInfo(CHOP_OutputInfo* info, const OP_Inputs* inputs, void*)
{
	// This CHOP is time sliced so we do not specify sample info
	info->numChannels = 1;
	return true;
}

void
TimeSliceGeneratorCHOP::getChannelName(int32_t index, OP_String *name, const OP_Inputs*, void*)
{
	name->setString("chan1");
}

void
TimeSliceGeneratorCHOP::execute(CHOP_Output* output,
							  const OP_Inputs* inputs,
							  void*)
{
	// Get all Parameters
	bool	applyScale = inputs->getParInt(APPLYSCALE_NAME) ? true : false;
	double	scale = applyScale ? inputs->getParDouble(SCALE_NAME) : 1.0;
	double	speed = inputs->getParDouble(SPEED_NAME);

	inputs->enablePar(SCALE_NAME, applyScale);

	// Menu items can be evaluated as either an integer menu position, or a string
	int		shape = inputs->getParInt(SHAPE_NAME);
	
	double	step = speed / output->sampleRate;
	double	value = 0;

	// Since this CHOP is time sliced numSamples is the number of frames
	// since we last cooked. Therefore, we output the value of those corresponding to those samples
	// which corresponds to value = fn(offset) where offset is the x value at that frame

	for (int i = 0; i < output->numSamples; i++)
	{
		switch (shape)
		{
			case Shape::Sine:
			{
				value = sin(myOffset * 2 * PI);
				break;
			}
			case Shape::Square:		
			{
				value = fmod(myOffset, 1.0) > 0.5 ? 1.0 : -1.0;
				break;
			}
			case Shape::Ramp:
			default:
			{
				value = fabs(fmod(myOffset, 1.0));
				break;
			}
		}

		value *= scale;
		myOffset += step;

		output->channels[0][i] = float(value);
	}
}

void
TimeSliceGeneratorCHOP::setupParameters(OP_ParameterManager* manager, void*)
{
	// shape
	{
		OP_StringParameter	sp;

		sp.name = SHAPE_NAME;
		sp.label = "Type";
		sp.page = "Generator";

		sp.defaultValue = "Sine";

		const char* names[] = { "Sine", "Square", "Ramp" };
		const char* labels[] = { "Sine", "Square", "Ramp" };

		OP_ParAppendResult res = manager->appendMenu(sp, 3, names, labels);
		assert(res == OP_ParAppendResult::Success);
	}

	// speed
	{
		OP_NumericParameter	np;

		np.name = SPEED_NAME;
		np.label = "Frequency";
		np.page = "Generator";
		np.defaultValues[0] = 1.0;
		np.minSliders[0] = -10.0;
		np.maxSliders[0] = 10.0;

		OP_ParAppendResult res = manager->appendFloat(np);
		assert(res == OP_ParAppendResult::Success);
	}

	// Apply Scale
	{
		OP_NumericParameter	np;

		np.name = APPLYSCALE_NAME;
		np.label = "Apply Scale";
		np.page = "Generator";
		np.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(np);
		assert(res == OP_ParAppendResult::Success);
	}

	// scale
	{
		OP_NumericParameter	np;

		np.name = SCALE_NAME;
		np.label = "Scale";
		np.page = "Generator";
		np.defaultValues[0] = 1.0;
		np.minSliders[0] = -10.0;
		np.maxSliders[0] = 10.0;

		OP_ParAppendResult res = manager->appendFloat(np);
		assert(res == OP_ParAppendResult::Success);
	}
}
