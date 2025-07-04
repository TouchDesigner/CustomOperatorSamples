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

CaseMenuItems
Parameters::evalCase(const TD::OP_Inputs* inputs)
{
	return static_cast<CaseMenuItems>(inputs->getParInt(CaseName));
}

bool
Parameters::evalKeepspaces(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(KeepspacesName) ? true : false;
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(TD::OP_ParameterManager* manager)
{
	{
		TD::OP_StringParameter p;
		p.name = CaseName;
		p.label = CaseLabel;
		p.page = "Filter";
		p.defaultValue = "Uppercamelcase";
		std::array<const char*, 3> Names =
		{
			"Uppercamelcase",
			"Lowercase",
			"Uppercase"
		};
		std::array<const char*, 3> Labels =
		{
			"Upper Camel Case",
			"Lower Case",
			"Upper Case"
		};
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = KeepspacesName;
		p.label = KeepspacesLabel;
		p.page = "Filter";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}


}

#pragma endregion