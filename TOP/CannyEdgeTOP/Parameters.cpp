#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

double
Parameters::evalLowthreshold(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(LowthresholdName);
}

double
Parameters::evalHighthreshold(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(HighthresholdName);
}

int
Parameters::evalApperturesize(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(ApperturesizeName);
}

bool
Parameters::evalL2gradient(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(L2gradientName) ? true : false;
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(TD::OP_ParameterManager* manager)
{
	{
		TD::OP_NumericParameter p;
		p.name = LowthresholdName;
		p.label = LowthresholdLabel;
		p.page = "Edge Detector";
		p.defaultValues[0] = 0.1;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 1.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = true;
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = HighthresholdName;
		p.label = HighthresholdLabel;
		p.page = "Edge Detector";
		p.defaultValues[0] = 0.9;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 1.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = true;
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = ApperturesizeName;
		p.label = ApperturesizeLabel;
		p.page = "Edge Detector";
		p.defaultValues[0] = 3;
		p.minSliders[0] = 1.0;
		p.maxSliders[0] = 31.0;
		p.minValues[0] = 1.0;
		p.maxValues[0] = 31.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = true;
		TD::OP_ParAppendResult res = manager->appendInt(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = L2gradientName;
		p.label = L2gradientLabel;
		p.page = "Edge Detector";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}


}

#pragma endregion