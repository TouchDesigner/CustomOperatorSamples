#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

using namespace TD;

ShapeMenuItems
Parameters::evalShape(const TD::OP_Inputs* input)
{
	return static_cast<ShapeMenuItems>(input->getParInt(ShapeName));
}

Color
Parameters::evalColor(const TD::OP_Inputs* input)
{
	std::array<double, 4> vals;
	input->getParDouble4(ColorName, vals[0], vals[1], vals[2], vals[3]);
	return Color((float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
}

bool
Parameters::evalGpudirect(const TD::OP_Inputs* input)
{
	return input->getParInt(GpudirectName) ? true : false;
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(TD::OP_ParameterManager* manager)
{
	{
		OP_StringParameter p;
		p.name = ShapeName;
		p.label = ShapeLabel;
		p.page = "Generator";
		p.defaultValue = "Cube";
		std::array<const char*, 4> Names =
		{
			"Point",
			"Line",
			"Square",
			"Cube"
		};
		std::array<const char*, 4> Labels =
		{
			"Point",
			"Line",
			"Square",
			"Cube"
		};
		OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = ColorName;
		p.label = ColorLabel;
		p.page = "Generator";
		
		const int ArraySize = 4;

		const std::array<double, ArraySize>  DefaultValues = { 1.0, 1.0, 1.0, 1.0 };
		const std::array<double, ArraySize>  MinSliders = { 0.0, 0.0, 0.0, 0.0 };
		const std::array<double, ArraySize>  MaxSliders = { 1.0, 1.0, 1.0, 1.0 };
		const std::array<double, ArraySize>  MinValues = { 0.0, 0.0, 0.0, 0.0 };
		const std::array<double, ArraySize>  MaxValues = { 1.0, 1.0, 1.0, 1.0 };
		const std::array<bool, ArraySize>  ClampMins = { false, false, false, false };
		const std::array<bool, ArraySize>  ClampMaxes = { false, false, false, false };
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

	{
		OP_NumericParameter p;
		p.name = GpudirectName;
		p.label = GpudirectLabel;
		p.page = "Generator";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}


}

#pragma endregion