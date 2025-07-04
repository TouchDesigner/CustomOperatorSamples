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

DistancetypeMenuItems
Parameters::evalDistancetype(const TD::OP_Inputs* inputs)
{
	return static_cast<DistancetypeMenuItems>(inputs->getParInt(DistancetypeName));
}

MasksizeMenuItems
Parameters::evalMasksize(const TD::OP_Inputs* inputs)
{
	return static_cast<MasksizeMenuItems>(inputs->getParInt(MasksizeName));
}

bool
Parameters::evalNormalize(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(NormalizeName) ? true : false;
}

ChannelMenuItems
Parameters::evalChannel(const TD::OP_Inputs* inputs)
{
	return static_cast<ChannelMenuItems>(inputs->getParInt(ChannelName));
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(TD::OP_ParameterManager* manager)
{
	{
		TD::OP_StringParameter p;
		p.name = DistancetypeName;
		p.label = DistancetypeLabel;
		p.page = "Transform";
		p.defaultValue = "L1";
		std::array<const char*, 3> Names =
		{
			"L1",
			"L2",
			"C"
		};
		std::array<const char*, 3> Labels =
		{
			"L1",
			"L2",
			"C"
		};
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_StringParameter p;
		p.name = MasksizeName;
		p.label = MasksizeLabel;
		p.page = "Transform";
		p.defaultValue = "Three";
		std::array<const char*, 3> Names =
		{
			"Three",
			"Five",
			"Precise"
		};
		std::array<const char*, 3> Labels =
		{
			"3x3",
			"5x5",
			"Precise"
		};
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = NormalizeName;
		p.label = NormalizeLabel;
		p.page = "Transform";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_StringParameter p;
		p.name = ChannelName;
		p.label = ChannelLabel;
		p.page = "Transform";
		p.defaultValue = "R";
		std::array<const char*, 4> Names =
		{
			"R",
			"G",
			"B",
			"A"
		};
		std::array<const char*, 4> Labels =
		{
			"R",
			"G",
			"B",
			"A"
		};
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}


}

#pragma endregion