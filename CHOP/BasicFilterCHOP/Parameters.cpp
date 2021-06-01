#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

bool
Parameters::evalApplyscale(const OP_Inputs* input)
{
	return input->getParInt(ApplyscaleName) ? true : false;
}

double
Parameters::evalScale(const OP_Inputs* input)
{
	return input->getParDouble(ScaleName);
}

bool
Parameters::evalApplyoffset(const OP_Inputs* input)
{
	return input->getParInt(ApplyoffsetName) ? true : false;
}

double
Parameters::evalOffset(const OP_Inputs* input)
{
	return input->getParDouble(OffsetName);
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(OP_ParameterManager* manager)
{
	{
		OP_NumericParameter p;
		p.name = ApplyscaleName;
		p.label = ApplyscaleLabel;
		p.page = "Filter";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
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
		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = ApplyoffsetName;
		p.label = ApplyoffsetLabel;
		p.page = "Filter";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
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
		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}


}

#pragma endregion