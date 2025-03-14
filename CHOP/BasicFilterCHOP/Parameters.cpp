#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

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

bool
Parameters::evalApplyoffset(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(ApplyoffsetName) ? true : false;
}

double
Parameters::evalOffset(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(OffsetName);
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(TD::OP_ParameterManager* manager)
{
	{
		TD::OP_NumericParameter p;
		p.name = ApplyscaleName;
		p.label = ApplyscaleLabel;
		p.page = "Filter";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = ScaleName;
		p.label = ScaleLabel;
		p.page = "Filter";
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
		p.name = ApplyoffsetName;
		p.label = ApplyoffsetLabel;
		p.page = "Filter";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = OffsetName;
		p.label = OffsetLabel;
		p.page = "Filter";
		p.defaultValues[0] = 0.0;
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