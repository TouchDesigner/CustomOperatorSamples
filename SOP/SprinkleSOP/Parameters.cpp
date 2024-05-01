#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

using namespace TD;

#pragma region Evals

double
Parameters::evalSeed(const OP_Inputs* input)
{
	return input->getParDouble(SeedName);
}

GenerateMenuItems
Parameters::evalGenerate(const OP_Inputs* input)
{
	return static_cast<GenerateMenuItems>(input->getParInt(GenerateName));
}

int
Parameters::evalPointcount(const OP_Inputs* input)
{
	return input->getParInt(PointcountName);
}

bool
Parameters::evalSeparatepoints(const OP_Inputs* input)
{
	return input->getParInt(SeparatepointsName) ? true : false;
}

double
Parameters::evalMinimumdistance(const OP_Inputs* input)
{
	return input->getParDouble(MinimumdistanceName);
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(OP_ParameterManager* manager)
{
	{
		OP_NumericParameter p;
		p.name = SeedName;
		p.label = SeedLabel;
		p.page = "Sprinkle";
		p.defaultValues[0] = 1.0;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter p;
		p.name = GenerateName;
		p.label = GenerateLabel;
		p.page = "Sprinkle";
		p.defaultValue = "Area";
		std::array<const char*, 4> Names =
		{
			"Area",
			"Primitive",
			"Boundingbox",
			"Volume"
		};
		std::array<const char*, 4> Labels =
		{
			"Surface Area",
			"Per Primitive",
			"Bounding Box",
			"Inside Volume"
		};
		OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = PointcountName;
		p.label = PointcountLabel;
		p.page = "Sprinkle";
		p.defaultValues[0] = 100;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 1000.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendInt(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = SeparatepointsName;
		p.label = SeparatepointsLabel;
		p.page = "Sprinkle";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = MinimumdistanceName;
		p.label = MinimumdistanceLabel;
		p.page = "Sprinkle";
		p.defaultValues[0] = 0.01;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 1.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}


}

#pragma endregion