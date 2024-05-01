#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

using namespace TD;

#pragma region Evals

const OP_CHOPInput*
Parameters::evalTranslatechop(const TD::OP_Inputs* input)
{
	return input->getParCHOP(TranslatechopName);
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(TD::OP_ParameterManager* manager)
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