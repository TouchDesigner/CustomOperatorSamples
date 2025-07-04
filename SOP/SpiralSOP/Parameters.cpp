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

OrientationMenuItems
Parameters::evalOrientation(const TD::OP_Inputs* inputs)
{
	return static_cast<OrientationMenuItems>(inputs->getParInt(OrientationName));
}

double
Parameters::evalTopradius(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(TopradiusName);
}

double
Parameters::evalBottomradius(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(BottomradiusName);
}

double
Parameters::evalHeight(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(HeightName);
}

double
Parameters::evalTurns(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(TurnsName);
}

int
Parameters::evalDivisions(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(DivisionsName);
}

OutputgeometryMenuItems
Parameters::evalOutputgeometry(const TD::OP_Inputs* inputs)
{
	return static_cast<OutputgeometryMenuItems>(inputs->getParInt(OutputgeometryName));
}

double
Parameters::evalStripwidth(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(StripwidthName);
}

bool
Parameters::evalGpudirect(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(GpudirectName) ? true : false;
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(TD::OP_ParameterManager* manager)
{
	{
		TD::OP_StringParameter p;
		p.name = OrientationName;
		p.label = OrientationLabel;
		p.page = "Spiral";
		p.defaultValue = "X";
		std::array<const char*, 3> Names =
		{
			"X",
			"Y",
			"Z"
		};
		std::array<const char*, 3> Labels =
		{
			"X",
			"Y",
			"Z"
		};
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = TopradiusName;
		p.label = TopradiusLabel;
		p.page = "Spiral";
		p.defaultValues[0] = 0.3;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = BottomradiusName;
		p.label = BottomradiusLabel;
		p.page = "Spiral";
		p.defaultValues[0] = 1.0;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = HeightName;
		p.label = HeightLabel;
		p.page = "Spiral";
		p.defaultValues[0] = 2.0;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 20.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = TurnsName;
		p.label = TurnsLabel;
		p.page = "Spiral";
		p.defaultValues[0] = 5.0;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = DivisionsName;
		p.label = DivisionsLabel;
		p.page = "Spiral";
		p.defaultValues[0] = 100;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 500.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendInt(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_StringParameter p;
		p.name = OutputgeometryName;
		p.label = OutputgeometryLabel;
		p.page = "Spiral";
		p.defaultValue = "Line";
		std::array<const char*, 3> Names =
		{
			"Points",
			"Line",
			"Trianglestrip"
		};
		std::array<const char*, 3> Labels =
		{
			"Points",
			"Line",
			"Triangle Strip"
		};
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = StripwidthName;
		p.label = StripwidthLabel;
		p.page = "Spiral";
		p.defaultValues[0] = 0.2;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 5.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = GpudirectName;
		p.label = GpudirectLabel;
		p.page = "Spiral";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}


}

#pragma endregion