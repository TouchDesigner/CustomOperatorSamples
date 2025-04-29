#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

CaseMenuItems
Parameters::evalCase(const TD::OP_Inputs* inputs)
{
	return static_cast<CaseMenuItems>(inputs->getParInt(CaseName));
}

bool
Parameters::evalKeepspaces(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(KeepspacesName) ? true : false;
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(TD::OP_ParameterManager* manager)
{
	{
		TD::OP_StringParameter p;
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
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = KeepspacesName;
		p.label = KeepspacesLabel;
		p.page = "Filter";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}


}

#pragma endregion