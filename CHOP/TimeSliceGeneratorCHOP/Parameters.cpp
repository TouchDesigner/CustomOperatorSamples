#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

TypeMenuItems
Parameters::evalType(const TD::OP_Inputs* inputs)
{
	return static_cast<TypeMenuItems>(inputs->getParInt(TypeName));
}

double
Parameters::evalFrequency(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(FrequencyName);
}

bool
Parameters::evalApplyscale(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(ApplyscaleName) ? true : false;
}

double
Parameters::evalScale(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(ScaleName);
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(TD::OP_ParameterManager* manager)
{
	{
		TD::OP_StringParameter p;
		p.name = TypeName;
		p.label = TypeLabel;
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
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = FrequencyName;
		p.label = FrequencyLabel;
		p.page = "Generator";
		p.defaultValues[0] = 1.0;
		p.minSliders[0] = -10.0;
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
		p.name = ApplyscaleName;
		p.label = ApplyscaleLabel;
		p.page = "Generator";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = ScaleName;
		p.label = ScaleLabel;
		p.page = "Generator";
		p.defaultValues[0] = 1.0;
		p.minSliders[0] = -10.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}


}

#pragma endregion