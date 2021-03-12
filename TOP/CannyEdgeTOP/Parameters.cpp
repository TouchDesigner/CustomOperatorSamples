#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

double
Parameters::evalLowthreshold(const OP_Inputs* input)
{
	return input->getParDouble(LowthresholdName);
}

double
Parameters::evalHighthreshold(const OP_Inputs* input)
{
	return input->getParDouble(HighthresholdName);
}

int
Parameters::evalApperturesize(const OP_Inputs* input)
{
	return input->getParInt(ApperturesizeName);
}

bool
Parameters::evalL2gradient(const OP_Inputs* input)
{
	return input->getParInt(L2gradientName) ? true : false;
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(OP_ParameterManager* manager)
{
	{
		OP_NumericParameter p;
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
		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
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
		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = ApperturesizeName;
		p.label = ApperturesizeLabel;
		p.page = "Edge Detector";
		p.defaultValues[0] = 3;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendInt(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = L2GradientName;
		p.label = L2GradientLabel;
		p.page = "Edge Detector";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}


}

#pragma endregion