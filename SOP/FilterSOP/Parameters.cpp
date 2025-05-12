#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

const TD::OP_CHOPInput*
Parameters::evalTranslatechop(const TD::OP_Inputs* inputs)
{
	return inputs->getParCHOP(TranslatechopName);
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(TD::OP_ParameterManager* manager)
{
	{
		TD::OP_StringParameter p;
		p.name = TranslatechopName;
		p.label = TranslatechopLabel;
		p.page = "Filter";
		p.defaultValue = "";
		TD::OP_ParAppendResult res = manager->appendCHOP(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}


}

#pragma endregion