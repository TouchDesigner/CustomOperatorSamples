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

#include "BasicGeneratorCHOP.h"

#include <string>
#include <cassert>
#include <cmath>

// Names of the parameters
constexpr static char SAMPLES_NAME[]	= "Length";
constexpr static char CHAN_NAME[]		= "Channels";
constexpr static char APPLYSCALE_NAME[]	= "Applyscale";
constexpr static char SCALE_NAME[]		= "Scale";
constexpr static char OPERATION_NAME[]	= "Operation";

enum class
Operation
{
	Add = 0,
	Multiply = 1,
	Power = 2
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
	customInfo.opType->setString("Generatorbasic");
	// English readable name
	customInfo.opLabel->setString("Generator Basic");
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
	return new BasicGeneratorCHOP(info);
}

DLLEXPORT
void
DestroyCHOPInstance(CHOP_CPlusPlusBase* instance)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the CHOP using that instance is deleted, or
	// if the CHOP loads a different DLL
	delete (BasicGeneratorCHOP*)instance;
}

};


BasicGeneratorCHOP::BasicGeneratorCHOP(const OP_NodeInfo*)
{

}

BasicGeneratorCHOP::~BasicGeneratorCHOP()
{

}

void
BasicGeneratorCHOP::getGeneralInfo(CHOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	// This chop doesn't need to cook every frame
	ginfo->cookEveryFrameIfAsked = false;

	// Note: To disable timeslicing you'll need to turn this off, as well as ensure that
	// getOutputInfo() returns true, and likely also set the info->numSamples to how many
	// samples you want to generate for this CHOP. Otherwise it'll take on length of the
	// input CHOP, which may be timesliced.
	ginfo->timeslice = false;
}

bool
BasicGeneratorCHOP::getOutputInfo(CHOP_OutputInfo* info, const OP_Inputs* inputs, void*)
{
	info->numChannels = inputs->getParInt(CHAN_NAME);
	info->numSamples = inputs->getParInt(SAMPLES_NAME);
	info->startIndex = 0;
	return true;
}

void
BasicGeneratorCHOP::getChannelName(int32_t index, OP_String *name, const OP_Inputs*, void*)
{
	std::string channelName = "chan" + std::to_string(index);
	name->setString(channelName.c_str());
}

void
BasicGeneratorCHOP::execute(CHOP_Output* output,
							  const OP_Inputs* inputs,
							  void*)
{
	// Get all Parameters
	bool	applyScale = inputs->getParInt(APPLYSCALE_NAME) ? true : false;
	double	scale = applyScale ? inputs->getParDouble(SCALE_NAME) : 1.0;
	int		operation = inputs->getParInt(OPERATION_NAME);

	inputs->enablePar(SCALE_NAME, applyScale);

	// Get length and channels from the output since they first need to be set to the current size
	int		length = output->numSamples;
	int		channels = output->numChannels;

	// Calculate scale*(channel operation sample)
	double	curValue;
	for (int i = 0; i < channels; ++i) 
	{
		for (int j = 0; j < length; ++j) 
		{
			switch (operation) 
			{
				case Operation::Add:
				{
					curValue = (double)i + j;
					break;
				}
				case Operation::Multiply:
				{
					curValue = (double)i * j;
					break;
				}
				case Operation::Power:
				default:
				{
					curValue = std::pow((double)i, (double)j);
					break;
				}
			}

			curValue *= scale;

			output->channels[i][j] = (float)curValue;
		}
	}
}

void
BasicGeneratorCHOP::setupParameters(OP_ParameterManager* manager, void*)
{
	// Sample
	{
		OP_NumericParameter	np;

		np.name = SAMPLES_NAME;
		np.label = "Length";
		np.page = "Generator Basic";
		np.defaultValues[0] = 10;
		np.minSliders[0] = 1;
		np.maxSliders[0] = 50;
		np.minValues[0] = 1;
		np.clampMins[0] = true;
		
		OP_ParAppendResult res = manager->appendInt(np);
		assert(res == OP_ParAppendResult::Success);
	}

	// Number of Channels
	{
		OP_NumericParameter	np;

		np.name = CHAN_NAME;
		np.label = "Number of Channels";
		np.page = "Generator Basic";
		np.defaultValues[0] = 10;
		np.minSliders[0] = 0;
		np.maxSliders[0] = 50;

		OP_ParAppendResult res = manager->appendInt(np);
		assert(res == OP_ParAppendResult::Success);
	}

	// Apply Scale
	{
		OP_NumericParameter	np;

		np.name = APPLYSCALE_NAME;
		np.label = "Apply Scale";
		np.page = "Generator Basic";
		np.defaultValues[0] = false;
		
		OP_ParAppendResult res = manager->appendToggle(np);
		assert(res == OP_ParAppendResult::Success);
	}

	// Scale
	{
		OP_NumericParameter	np;

		np.name = SCALE_NAME;
		np.label = "Scale";
		np.page = "Generator Basic";
		np.defaultValues[0] = 1.0;
		np.minSliders[0] = -10.0;
		np.maxSliders[0] = 10.0;

		OP_ParAppendResult res = manager->appendFloat(np);
		assert(res == OP_ParAppendResult::Success);
	}

	// Operation
	{
		OP_StringParameter	sp;

		sp.name = OPERATION_NAME;
		sp.label = "Operation";
		sp.page = "Generator Basic";
		sp.defaultValue = "Add";
		
		const char* names[] = { "Add", "Multiply", "Power" };
		const char* labels[] = { "Add", "Multiply", "Power" };

		OP_ParAppendResult res = manager->appendMenu(sp, 3, names, labels);
		assert(res == OP_ParAppendResult::Success);
	}
}
