#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"


#pragma region Evals

OperationMenuItems
Parameters::evalOperation(const TD::OP_Inputs* input)
{
	return static_cast<OperationMenuItems>(input->getParInt(OperationName));
}

int
Parameters::evalReset(const TD::OP_Inputs* input)
{
	return input->getParInt(ResetName);
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(TD::OP_ParameterManager* manager)
{
	{
		OP_StringParameter p;
		p.name = OperationName;
		p.label = OperationLabel;
		p.page = "Filter";
		p.defaultValue = "Max";
		std::array<const char*, 3> Names =
		{
			"Max",
			"Min",
			"Average"
		};
		std::array<const char*, 3> Labels =
		{
			"Max",
			"Min",
			"Average"
		};
		OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = ResetName;
		p.label = ResetLabel;
		p.page = "Filter";
		OP_ParAppendResult res = manager->appendPulse(p);

		assert(res == OP_ParAppendResult::Success);
	}


}

#pragma endregion