#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

CaseMenuItems
Parameters::evalCase(const OP_Inputs* input)
{
	return static_cast<CaseMenuItems>(input->getParInt(CaseName));
}

bool
Parameters::evalKeepspaces(const OP_Inputs* input)
{
	return input->getParInt(KeepspacesName) ? true : false;
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(OP_ParameterManager* manager)
{
	{
		OP_StringParameter p;
		p.name = CaseName;
		p.label = CaseLabel;
		p.page = "Filter";
		p.defaultValue = "Uppercamelcase";
		std::array<const char*, 3> Names =
		{
			"Uppercamelcase",
			"Lowercase",
			"Uppercase"
		};
		std::array<const char*, 3> Labels =
		{
			"Upper Camel Case",
			"Lower Case",
			"Upper Case"
		};
		OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = KeepspacesName;
		p.label = KeepspacesLabel;
		p.page = "Filter";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}


}

#pragma endregion