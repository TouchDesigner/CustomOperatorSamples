#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

double
Parameters::evalBrightness(const OP_Inputs* input)
{
	return input->getParDouble(BrightnessName);
}

double
Parameters::evalSpeed(const OP_Inputs* input)
{
	return input->getParDouble(SpeedName);
}

int
Parameters::evalReset(const OP_Inputs* input)
{
	return input->getParInt(ResetName);
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(OP_ParameterManager* manager)
{
	{
		OP_NumericParameter p;
		p.name = BrightnessName;
		p.label = BrightnessLabel;
		p.page = "CPU Memory";
		p.defaultValues[0] = 1.0;
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
		p.name = SpeedName;
		p.label = SpeedLabel;
		p.page = "CPU Memory";
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
		p.name = ResetName;
		p.label = ResetLabel;
		p.page = "CPU Memory";
		OP_ParAppendResult res = manager->appendPulse(p);

		assert(res == OP_ParAppendResult::Success);
	}


}

#pragma endregion