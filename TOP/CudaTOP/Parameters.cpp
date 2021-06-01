#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

Color
Parameters::evalColor(const OP_Inputs* input)
{
	std::array<double, 4> vals;
	input->getParDouble4(ColorName, vals[0], vals[1], vals[2], vals[3]);
	return Color((float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(OP_ParameterManager* manager)
{
	{
		OP_NumericParameter p;
		p.name = ColorName;
		p.label = ColorLabel;
		p.page = "Generator";
		
		const int ArraySize = 4;

		const std::array<double, ArraySize>  DefaultValues = { 0.0, 0.0, 0.0, 0.0 };
		const std::array<double, ArraySize>  MinSliders = { 0.0, 0.0, 0.0, 0.0 };
		const std::array<double, ArraySize>  MaxSliders = { 1.0, 1.0, 1.0, 1.0 };
		const std::array<double, ArraySize>  MinValues = { 0.0, 0.0, 0.0, 0.0 };
		const std::array<double, ArraySize>  MaxValues = { 1.0, 1.0, 1.0, 1.0 };
		const std::array<bool, ArraySize>  ClampMins = { true, true, true, true };
		const std::array<bool, ArraySize>  ClampMaxes = { true, true, true, true };
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
		OP_ParAppendResult res = manager->appendRGBA(p);

		assert(res == OP_ParAppendResult::Success);
	}


}

#pragma endregion