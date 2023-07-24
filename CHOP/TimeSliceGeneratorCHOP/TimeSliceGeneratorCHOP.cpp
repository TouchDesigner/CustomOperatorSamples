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
#include <array>
#include <string>

static constexpr double PI = 3.141592653589793238463;

enum class TypeMenuItems
{
	Sine,
	Square,
	Ramp
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
	customInfo.authorName->setString("Author Name");
	customInfo.authorEmail->setString("email@email");

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
	bool	applyScale = inputs->getParInt("Applyscale") ? true : false;
	double	scale = inputs->getParDouble("Scale");
	double	speed = inputs->getParDouble("Frequency");

	inputs->enablePar("Scale", applyScale);

	// Menu items can be evaluated as either an integer menu position, or a string
	TypeMenuItems		shape = static_cast<TypeMenuItems>(inputs->getParInt("Type"));
	
	double	step = speed / output->sampleRate;
	double	value = 0;

	// Since this CHOP is time sliced numSamples is the number of frames
	// since we last cooked. Therefore, we output the value of those corresponding to those samples
	// which corresponds to value = fn(offset) where offset is the x value at that frame

	for (int i = 0; i < output->numSamples; i++)
	{
		switch (shape)
		{
			case TypeMenuItems::Sine:
			{
				value = sin(myOffset * 2 * PI);
				break;
			}
			case TypeMenuItems::Square:
			{
				value = fmod(myOffset, 1.0) > 0.5 ? 1.0 : -1.0;
				break;
			}
			case TypeMenuItems::Ramp:
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
	{
		OP_StringParameter p;
		p.name = "Type";
		p.label = "Type";
		p.page = "Generator";
		p.defaultValue = "Sine";
		std::array<const char*, 3> Names =
		{
			"Sine",
			"Square",
			"Ramp"
		};
		std::array<const char*, 3> Labels =
		{
			"Sine",
			"Square",
			"Ramp"
		};
		OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = "Frequency";
		p.label = "Frequency";
		p.page = "Generator";
		p.defaultValues[0] = 1.0;
		p.minSliders[0] = -10.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = "Applyscale";
		p.label = "Apply Scale";
		p.page = "Generator";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = "Scale";
		p.label = "Scale";
		p.page = "Generator";
		p.defaultValues[0] = 1.0;
		p.minSliders[0] = -10.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}
}
