#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

RaysMenuItems
Parameters::evalRays(const TD::OP_Inputs* inputs)
{
	return static_cast<RaysMenuItems>(inputs->getParInt(RaysName));
}

std::array<double, 3>
Parameters::evalDirection(const TD::OP_Inputs* inputs)
{
	std::array<double, 3> vals;
	inputs->getParDouble3(DirectionName, vals[0], vals[1], vals[2]);
	return vals;
}

std::array<double, 3>
Parameters::evalDestination(const TD::OP_Inputs* inputs)
{
	std::array<double, 3> vals;
	inputs->getParDouble3(DestinationName, vals[0], vals[1], vals[2]);
	return vals;
}

bool
Parameters::evalReverse(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(ReverseName) ? true : false;
}

TD::Color
Parameters::evalHitcolor(const TD::OP_Inputs* inputs)
{
	std::array<double, 4> vals;
	inputs->getParDouble4(HitcolorName, vals[0], vals[1], vals[2], vals[3]);
	return TD::Color((float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
}

TD::Color
Parameters::evalMisscolor(const TD::OP_Inputs* inputs)
{
	std::array<double, 4> vals;
	inputs->getParDouble4(MisscolorName, vals[0], vals[1], vals[2], vals[3]);
	return TD::Color((float)vals[0], (float)vals[1], (float)vals[2], (float)vals[3]);
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
		p.name = RaysName;
		p.label = RaysLabel;
		p.page = "Cast Ray";
		p.defaultValue = "Parallel";
		std::array<const char*, 2> Names =
		{
			"Parallel",
			"Radial"
		};
		std::array<const char*, 2> Labels =
		{
			"Parallel",
			"Radial"
		};
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = DirectionName;
		p.label = DirectionLabel;
		p.page = "Cast Ray";
		
		const int ArraySize = 3;

		const std::array<double, ArraySize>  DefaultValues = { 1.0, 0.0, 0.0 };
		const std::array<double, ArraySize>  MinSliders = { 0.0, 0.0, 0.0 };
		const std::array<double, ArraySize>  MaxSliders = { 1.0, 1.0, 1.0 };
		const std::array<double, ArraySize>  MinValues = { 0.0, 0.0, 0.0 };
		const std::array<double, ArraySize>  MaxValues = { 1.0, 1.0, 1.0 };
		const std::array<bool, ArraySize>  ClampMins = { false, false, false };
		const std::array<bool, ArraySize>  ClampMaxes = { false, false, false };
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
		TD::OP_ParAppendResult res = manager->appendXYZ(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = DestinationName;
		p.label = DestinationLabel;
		p.page = "Cast Ray";
		
		const int ArraySize = 3;

		const std::array<double, ArraySize>  DefaultValues = { 0.0, 0.0, 0.0 };
		const std::array<double, ArraySize>  MinSliders = { 0.0, 0.0, 0.0 };
		const std::array<double, ArraySize>  MaxSliders = { 1.0, 1.0, 1.0 };
		const std::array<double, ArraySize>  MinValues = { 0.0, 0.0, 0.0 };
		const std::array<double, ArraySize>  MaxValues = { 1.0, 1.0, 1.0 };
		const std::array<bool, ArraySize>  ClampMins = { false, false, false };
		const std::array<bool, ArraySize>  ClampMaxes = { false, false, false };
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
		TD::OP_ParAppendResult res = manager->appendXYZ(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = ReverseName;
		p.label = ReverseLabel;
		p.page = "Cast Ray";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = HitcolorName;
		p.label = HitcolorLabel;
		p.page = "Cast Ray";
		
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
		TD::OP_ParAppendResult res = manager->appendRGBA(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = MisscolorName;
		p.label = MisscolorLabel;
		p.page = "Cast Ray";
		
		const int ArraySize = 4;

		const std::array<double, ArraySize>  DefaultValues = { 0.0, 0.0, 0.0, 1.0 };
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
		TD::OP_ParAppendResult res = manager->appendRGBA(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = ScaleName;
		p.label = ScaleLabel;
		p.page = "Cast Ray";
		p.defaultValues[0] = 1.0;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 1.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}


}

#pragma endregion