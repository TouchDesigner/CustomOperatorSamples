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

double
Parameters::evalSeed(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(SeedName);
}

GenerateMenuItems
Parameters::evalGenerate(const TD::OP_Inputs* inputs)
{
	return static_cast<GenerateMenuItems>(inputs->getParInt(GenerateName));
}

int
Parameters::evalPointcount(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(PointcountName);
}

bool
Parameters::evalSeparatepoints(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(SeparatepointsName) ? true : false;
}

double
Parameters::evalMinimumdistance(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(MinimumdistanceName);
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(TD::OP_ParameterManager* manager)
{
	{
		TD::OP_NumericParameter p;
		p.name = SeedName;
		p.label = SeedLabel;
		p.page = "Sprinkle";
		p.defaultValues[0] = 1.0;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_StringParameter p;
		p.name = GenerateName;
		p.label = GenerateLabel;
		p.page = "Sprinkle";
		p.defaultValue = "Area";
		std::array<const char*, 4> Names =
		{
			"Area",
			"Primitive",
			"Boundingbox",
			"Volume"
		};
		std::array<const char*, 4> Labels =
		{
			"Surface Area",
			"Per Primitive",
			"Bounding Box",
			"Inside Volume"
		};
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = PointcountName;
		p.label = PointcountLabel;
		p.page = "Sprinkle";
		p.defaultValues[0] = 100;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 1000.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendInt(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = SeparatepointsName;
		p.label = SeparatepointsLabel;
		p.page = "Sprinkle";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = MinimumdistanceName;
		p.label = MinimumdistanceLabel;
		p.page = "Sprinkle";
		p.defaultValues[0] = 0.01;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 1.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}


}

#pragma endregion