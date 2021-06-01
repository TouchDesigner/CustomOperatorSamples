#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

const OP_CHOPInput*
Parameters::evalTranslatechop(const OP_Inputs* input)
{
	return input->getParCHOP(TranslatechopName);
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(OP_ParameterManager* manager)
{
	{
		OP_StringParameter p;
		p.name = TranslatechopName;
		p.label = TranslatechopLabel;
		p.page = "Filter";
		p.defaultValue = "";
		OP_ParAppendResult res = manager->appendCHOP(p);

		assert(res == OP_ParAppendResult::Success);
	}


}

#pragma endregion