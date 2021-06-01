#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

Color
Parameters::evalColor1(const OP_Inputs* input)
{
	std::array<double, 3> vals;
	input->getParDouble3(Color1Name, vals[0], vals[1], vals[2]);
	return Color((float)vals[0], (float)vals[1], (float)vals[2], 1.0f);
}

Color
Parameters::evalColor2(const OP_Inputs* input)
{
	std::array<double, 3> vals;
	input->getParDouble3(Color2Name, vals[0], vals[1], vals[2]);
	return Color((float)vals[0], (float)vals[1], (float)vals[2], 1.0f);
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
		p.name = Color1Name;
		p.label = Color1Label;
		p.page = "OpenGL";
		
		const int ArraySize = 3;

		const std::array<double, ArraySize>  DefaultValues = { 1.0, 0.5, 0.8 };
		const std::array<double, ArraySize>  MinSliders = { 0.0, 0.0, 0.0 };
		const std::array<double, ArraySize>  MaxSliders = { 1.0, 1.0, 1.0 };
		const std::array<double, ArraySize>  MinValues = { 0.0, 0.0, 0.0 };
		const std::array<double, ArraySize>  MaxValues = { 1.0, 1.0, 1.0 };
		const std::array<bool, ArraySize>  ClampMins = { true, true, true };
		const std::array<bool, ArraySize>  ClampMaxes = { true, true, true };
		for (int i = 0; i < DefaultValues.size(); ++i)
		{
			p.defaultValues[i] = DefaultValues[i];
			p.minSliders[i] = MinSliders[i];
			p.maxSliders[i] = MaxSliders[i];
			p.minValues[i] = MinValues[i];
			p.maxValues[i] = MaxValues[i];
			p.clampMins[i] = ClampMins[i];
			p.clampMaxes[i] = ClampMaxes[i];
		}
		OP_ParAppendResult res = manager->appendRGB(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = Color2Name;
		p.label = Color2Label;
		p.page = "OpenGL";
		
		const int ArraySize = 3;

		const std::array<double, ArraySize>  DefaultValues = { 1.0, 1.0, 0.25 };
		const std::array<double, ArraySize>  MinSliders = { 0.0, 0.0, 0.0 };
		const std::array<double, ArraySize>  MaxSliders = { 1.0, 1.0, 1.0 };
		const std::array<double, ArraySize>  MinValues = { 0.0, 0.0, 0.0 };
		const std::array<double, ArraySize>  MaxValues = { 1.0, 1.0, 1.0 };
		const std::array<bool, ArraySize>  ClampMins = { true, true, true };
		const std::array<bool, ArraySize>  ClampMaxes = { true, true, true };
		for (int i = 0; i < DefaultValues.size(); ++i)
		{
			p.defaultValues[i] = DefaultValues[i];
			p.minSliders[i] = MinSliders[i];
			p.maxSliders[i] = MaxSliders[i];
			p.minValues[i] = MinValues[i];
			p.maxValues[i] = MaxValues[i];
			p.clampMins[i] = ClampMins[i];
			p.clampMaxes[i] = ClampMaxes[i];
		}
		OP_ParAppendResult res = manager->appendRGB(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = SpeedName;
		p.label = SpeedLabel;
		p.page = "OpenGL";
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
		p.page = "OpenGL";
		OP_ParAppendResult res = manager->appendPulse(p);

		assert(res == OP_ParAppendResult::Success);
	}


}

#pragma endregion