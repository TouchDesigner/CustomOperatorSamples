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

// Parameters.cpp generated using the cppParsTemplateGen Palette Component.
// https://derivative.ca/UserGuide/Palette:cppParsTemplateGen

#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

OperationMenuItems
Parameters::evalOperation(const TD::OP_Inputs* inputs)
{
	return static_cast<OperationMenuItems>(inputs->getParInt(OperationName));
}

int
Parameters::evalReset(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(ResetName);
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(TD::OP_ParameterManager* manager)
{
	{
		TD::OP_StringParameter p;
		p.name = OperationName;
		p.label = OperationLabel;
		p.page = "Filter";
		p.defaultValue = "Max";
		std::array<const char*, 3> Names =
		{
			"Max",
			"Min",
			"Average"
		};
		std::array<const char*, 3> Labels =
		{
			"Max",
			"Min",
			"Average"
		};
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = ResetName;
		p.label = ResetLabel;
		p.page = "Filter";
		TD::OP_ParAppendResult res = manager->appendPulse(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}


}

#pragma endregion