#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

ShapeMenuItems
Parameters::evalShape(const OP_Inputs* input)
{
	return static_cast<ShapeMenuItems>(input->getParInt(ShapeName));
}

Color
Parameters::evalColor(const OP_Inputs* input)
{
	std::array<double, 4> vals;
	input->getParDouble4(ColorName, vals[0], vals[1], vals[2], vals[3]);
	return Color((float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
}

bool
Parameters::evalGpudirect(const OP_Inputs* input)
{
	return input->getParInt(GpudirectName) ? true : false;
}

double
Parameters::evalScale(const OP_Inputs* input)
{
	return input->getParDouble(ScaleName);
}

const OP_CHOPInput*
Parameters::evalPointschop(const OP_Inputs* input)
{
	return input->getParCHOP(PointsChopName);
}




#pragma endregion

#pragma region Setup

void
Parameters::setup(OP_ParameterManager* manager)
{
	{
		OP_StringParameter p;
		p.name = PointsChopName;
		p.label = PointsChopLabel;
		p.page = "Generator";
		p.defaultValue = "";

		OP_ParAppendResult res = manager->appendCHOP(p);

		assert(res == OP_ParAppendResult::Success);
	}
	
	{
		OP_StringParameter p;
		p.name = ShapeName;
		p.label = ShapeLabel;
		p.page = "Generator";
		p.defaultValue = "KDTree";
		std::array<const char*, 6> Names =
		{
			"Point",
			"Line",
			"Square",
			"Cube",
			"Voronoi",
			"KDTree"
		};
		std::array<const char*, 6> Labels =
		{
			"Point",
			"Line",
			"Square",
			"Cube",
			"Voronoi",
			"KDTree"
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

	{
		OP_NumericParameter p;
		p.name = ScaleName;
		p.label = ScaleLabel;
		p.page = "Generator";
		p.defaultValues[0] = 1.0;

		p.defaultValues[0] = 1.0;
		p.minSliders[0] = 0.01;
		p.maxSliders[0] = 1.0;
		p.minValues[0] = 0.01;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = true;

		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}


}

#pragma endregion