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

#include "TimeSliceFilterCHOP.h"

#include <cassert>
#include <string>

// Names of the parameters
constexpr static char OPERATION_NAME[]	= "Speed";
constexpr static char RESET_NAME[]		= "Applyscale";

enum class
Operation
{
	Max = 0,
	Min = 1,
	Average = 2
};

class FilterValues
{
public:
	FilterValues() : 
		myMax{}, myMin{}, myAverage{}, myN{}, myValid(false)
	{

	}

	void 
	addValue(double value) 
	{
		if (!myValid) 
		{
			myMax = myMin = myAverage = value;
			myN = 1;
			myValid = true;
		}
		else 
		{
			if (value > myMax)
			{
				myMax = value;
			}
			if (value < myMin)
			{
				myMin = value;
			}
			myAverage = (myAverage * myN + value) / ((double)myN + 1);
			myN++;
		}
	}

	double 
	get(int op)
	{
		switch (op)
		{
			case Operation::Min:
			{
				return myMin;
			}
			case Operation::Max:
			{
				return myMax;
			}
			case Operation::Average:
			default:
			{
				return myAverage;
			}
		}
	}

private:
	double myMax;
	double myMin;
	double myAverage;
	int myN;
	bool myValid;
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
	customInfo.opType->setString("Timeslicefilter");
	// English readable name
	customInfo.opLabel->setString("Time Slice Filter");
	// Information of the author of the node
	customInfo.authorName->setString("Gabriel Robels");
	customInfo.authorEmail->setString("support@derivative.ca");

	// This CHOP takes no inputs
	customInfo.minInputs = 1;
	customInfo.maxInputs = 20;
}

DLLEXPORT
CHOP_CPlusPlusBase*
CreateCHOPInstance(const OP_NodeInfo* info)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per CHOP that is using the .dll
	return new TimeSliceFilterCHOP(info);
}

DLLEXPORT
void
DestroyCHOPInstance(CHOP_CPlusPlusBase* instance)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the CHOP using that instance is deleted, or
	// if the CHOP loads a different DLL
	delete (TimeSliceFilterCHOP*)instance;
}

};


TimeSliceFilterCHOP::TimeSliceFilterCHOP(const OP_NodeInfo*) : 
	myValues()
{

}

TimeSliceFilterCHOP::~TimeSliceFilterCHOP()
{

}

void
TimeSliceFilterCHOP::getGeneralInfo(CHOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	// This will cause the node to cook every frame if the output is used
	ginfo->cookEveryFrame = true;

	// Note: To disable timeslicing you'll need to turn this off, as well as ensure that
	// getOutputInfo() returns true, and likely also set the info->numSamples to how many
	// samples you want to generate for this CHOP. Otherwise it'll take on length of the
	// input CHOP, which may be timesliced.
	ginfo->timeslice = true;
}

bool
TimeSliceFilterCHOP::getOutputInfo(CHOP_OutputInfo* info, const OP_Inputs* inputs, void*)
{
	// This CHOP is time sliced so we do not specify sample info

	// Create as many channels to filter all channels of all inputs
	int numInputs = inputs->getNumInputs();
	// Since inputs connected might be out of order we need to loop until we get as many inputs as numInputs
	int index = 0;
	int inputsGotten = 0;
	int totalNumChannels = 0;

	while (inputsGotten < numInputs)
	{
		const OP_CHOPInput* input = inputs->getInputCHOP(index);
		if (input)
		{
			totalNumChannels += input->numChannels;
			inputsGotten++;
		}
		index++;
	}
	info->numChannels = totalNumChannels;
	myValues.resize(totalNumChannels);

	return true;
}

void
TimeSliceFilterCHOP::getChannelName(int32_t index, OP_String *name, const OP_Inputs*, void*)
{
	std::string channelName = "chan" + std::to_string(index + 1);
	name->setString(channelName.c_str());
}

void
TimeSliceFilterCHOP::execute(CHOP_Output* output,
							  const OP_Inputs* inputs,
							  void*)
{
	int operation = inputs->getParInt(OPERATION_NAME);

	int numInputs = inputs->getNumInputs();
	// Since inputs connected might be out of order we need to loop until we get as many inputs as numInputs
	int index = 0;
	int inputsGotten = 0;
	int channelIndex = 0;

	while (inputsGotten < numInputs)
	{
		const OP_CHOPInput* input = inputs->getInputCHOP(index);
		if (input)
		{
			for (int i = 0; i < input->numChannels; ++i) 
			{
				for (int j = 0; j < input->numSamples; ++j)
				{
					myValues.at(channelIndex).addValue(input->getChannelData(i)[j]);
				}
				for (int j = 0; j < output->numSamples; ++j)
				{
					output->channels[channelIndex][j] = (float)myValues.at(channelIndex).get(operation);
				}
				channelIndex++;
			}
			inputsGotten++;
		}
		index++;
	}
}

void
TimeSliceFilterCHOP::setupParameters(OP_ParameterManager* manager, void*)
{
	// Operation
	{
		OP_StringParameter	sp;

		sp.name = OPERATION_NAME;
		sp.label = "Operation";
		sp.page = "Filter";

		sp.defaultValue = "Max";

		const char* names[] = { "Max", "Min", "Average" };
		const char* labels[] = { "Max", "Min", "Average" };

		OP_ParAppendResult res = manager->appendMenu(sp, 3, names, labels);
		assert(res == OP_ParAppendResult::Success);
	}

	// Reset
	{
		OP_NumericParameter	np;

		np.name = RESET_NAME;
		np.label = "Reset";
		np.page = "Filter";

		OP_ParAppendResult res = manager->appendPulse(np);
		assert(res == OP_ParAppendResult::Success);
	}
}

void
TimeSliceFilterCHOP::pulsePressed(const char* name, void*)
{
	if (!strcmp(name, RESET_NAME))
	{
		for (FilterValues& value : myValues)
		{
			value = FilterValues();
		}
	}
}
