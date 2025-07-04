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

int
Parameters::evalNumlevels(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(NumlevelsName);
}

double
Parameters::evalPyramidscale(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(PyramidscaleName);
}

int
Parameters::evalWindowsize(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(WindowsizeName);
}

int
Parameters::evalIterations(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(IterationsName);
}

int
Parameters::evalPolyn(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(PolynName);
}

double
Parameters::evalPolysigma(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(PolysigmaName);
}

bool
Parameters::evalUsegaussianfilter(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(UsegaussianfilterName) ? true : false;
}

bool
Parameters::evalUsepreviousflow(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(UsepreviousflowName) ? true : false;
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
		TD::OP_NumericParameter p;
		p.name = NumlevelsName;
		p.label = NumlevelsLabel;
		p.page = "Optical Flow";
		p.defaultValues[0] = 5;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendInt(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = PyramidscaleName;
		p.label = PyramidscaleLabel;
		p.page = "Optical Flow";
		p.defaultValues[0] = 0.5;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 0.5;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 0.5;
		p.clampMins[0] = true;
		p.clampMaxes[0] = true;
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = WindowsizeName;
		p.label = WindowsizeLabel;
		p.page = "Optical Flow";
		p.defaultValues[0] = 13;
		p.minSliders[0] = 1.0;
		p.maxSliders[0] = 100.0;
		p.minValues[0] = 1.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendInt(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = IterationsName;
		p.label = IterationsLabel;
		p.page = "Optical Flow";
		p.defaultValues[0] = 10;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 50.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendInt(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = PolynName;
		p.label = PolynLabel;
		p.page = "Optical Flow";
		p.defaultValues[0] = 5;
		p.minSliders[0] = 5.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 5.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendInt(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = PolysigmaName;
		p.label = PolysigmaLabel;
		p.page = "Optical Flow";
		p.defaultValues[0] = 1.1;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 2.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = UsegaussianfilterName;
		p.label = UsegaussianfilterLabel;
		p.page = "Optical Flow";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = UsepreviousflowName;
		p.label = UsepreviousflowLabel;
		p.page = "Optical Flow";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_StringParameter p;
		p.name = ChannelName;
		p.label = ChannelLabel;
		p.page = "Optical Flow";
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