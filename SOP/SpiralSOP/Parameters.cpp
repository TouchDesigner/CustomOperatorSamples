#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

using namespace TD;

#pragma region Evals

OrientationMenuItems
Parameters::evalOrientation(const TD::OP_Inputs* input)
{
	return static_cast<OrientationMenuItems>(input->getParInt(OrientationName));
}

double
Parameters::evalTopradius(const TD::OP_Inputs* input)
{
	return input->getParDouble(TopradiusName);
}

double
Parameters::evalBottomradius(const TD::OP_Inputs* input)
{
	return input->getParDouble(BottomradiusName);
}

double
Parameters::evalHeight(const TD::OP_Inputs* input)
{
	return input->getParDouble(HeightName);
}

double
Parameters::evalTurns(const TD::OP_Inputs* input)
{
	return input->getParDouble(TurnsName);
}

int
Parameters::evalDivisions(const TD::OP_Inputs* input)
{
	return input->getParInt(DivisionsName);
}

OutputgeometryMenuItems
Parameters::evalOutputgeometry(const TD::OP_Inputs* input)
{
	return static_cast<OutputgeometryMenuItems>(input->getParInt(OutputgeometryName));
}

double
Parameters::evalStripwidth(const TD::OP_Inputs* input)
{
	return input->getParDouble(StripwidthName);
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
		p.name = OrientationName;
		p.label = OrientationLabel;
		p.page = "Spiral";
		p.defaultValue = "X";
		std::array<const char*, 3> Names =
		{
			"X",
			"Y",
			"Z"
		};
		std::array<const char*, 3> Labels =
		{
			"X",
			"Y",
			"Z"
		};
		OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = TopradiusName;
		p.label = TopradiusLabel;
		p.page = "Spiral";
		p.defaultValues[0] = 0.3;
		p.minSliders[0] = 0.0;
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
		p.name = BottomradiusName;
		p.label = BottomradiusLabel;
		p.page = "Spiral";
		p.defaultValues[0] = 1.0;
		p.minSliders[0] = 0.0;
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
		p.name = HeightName;
		p.label = HeightLabel;
		p.page = "Spiral";
		p.defaultValues[0] = 2.0;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 20.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = TurnsName;
		p.label = TurnsLabel;
		p.page = "Spiral";
		p.defaultValues[0] = 5.0;
		p.minSliders[0] = 0.0;
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
		p.name = DivisionsName;
		p.label = DivisionsLabel;
		p.page = "Spiral";
		p.defaultValues[0] = 100;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 500.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendInt(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter p;
		p.name = OutputgeometryName;
		p.label = OutputgeometryLabel;
		p.page = "Spiral";
		p.defaultValue = "Line";
		std::array<const char*, 3> Names =
		{
			"Points",
			"Line",
			"Trianglestrip"
		};
		std::array<const char*, 3> Labels =
		{
			"Points",
			"Line",
			"Triangle Strip"
		};
		OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = StripwidthName;
		p.label = StripwidthLabel;
		p.page = "Spiral";
		p.defaultValues[0] = 0.2;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 5.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = GpudirectName;
		p.label = GpudirectLabel;
		p.page = "Spiral";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}


}

#pragma endregion