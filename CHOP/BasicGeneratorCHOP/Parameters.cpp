#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

int
Parameters::evalLength(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(LengthName);
}

int
Parameters::evalNumberofchannels(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(NumberofchannelsName);
}

bool
Parameters::evalApplyscale(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(ApplyscaleName) ? true : false;
}

double
Parameters::evalScale(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(ScaleName);
}

OperationMenuItems
Parameters::evalOperation(const TD::OP_Inputs* inputs)
{
	return static_cast<OperationMenuItems>(inputs->getParInt(OperationName));
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(TD::OP_ParameterManager* manager)
{
	{
		TD::OP_NumericParameter p;
		p.name = LengthName;
		p.label = LengthLabel;
		p.page = "Generator Basic";
		p.defaultValues[0] = 10;
		p.minSliders[0] = 1.0;
		p.maxSliders[0] = 50.0;
		p.minValues[0] = 1.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendInt(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = NumberofchannelsName;
		p.label = NumberofchannelsLabel;
		p.page = "Generator Basic";
		p.defaultValues[0] = 10;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 50.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendInt(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = ApplyscaleName;
		p.label = ApplyscaleLabel;
		p.page = "Generator Basic";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = ScaleName;
		p.label = ScaleLabel;
		p.page = "Generator Basic";
		p.defaultValues[0] = 1.0;
		p.minSliders[0] = -10.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_StringParameter p;
		p.name = OperationName;
		p.label = OperationLabel;
		p.page = "Generator Basic";
		p.defaultValue = "Add";
		std::array<const char*, 3> Names =
		{
			"Add",
			"Multiply",
			"Power"
		};
		std::array<const char*, 3> Labels =
		{
			"Add",
			"Multiply",
			"Power"
		};
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}


}

#pragma endregion