#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

double
Parameters::evalSeed(const OP_Inputs* input)
{
	return input->getParDouble(SeedName);
}

int
Parameters::evalRows(const OP_Inputs* input)
{
	return input->getParInt(RowsName);
}

int
Parameters::evalColumns(const OP_Inputs* input)
{
	return input->getParInt(ColumnsName);
}

int
Parameters::evalLength(const OP_Inputs* input)
{
	return input->getParInt(LengthName);
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
		p.page = "Generator";
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
		p.name = RowsName;
		p.label = RowsLabel;
		p.page = "Generator";
		p.defaultValues[0] = 3;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 100.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendInt(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = ColumnsName;
		p.label = ColumnsLabel;
		p.page = "Generator";
		p.defaultValues[0] = 4;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 100.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendInt(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = LengthName;
		p.label = LengthLabel;
		p.page = "Generator";
		p.defaultValues[0] = 5;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 100.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendInt(p);

		assert(res == OP_ParAppendResult::Success);
	}


}

#pragma endregion