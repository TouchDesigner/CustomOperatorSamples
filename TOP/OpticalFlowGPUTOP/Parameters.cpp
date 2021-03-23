#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

int
Parameters::evalNumlevels(const OP_Inputs* input)
{
	return input->getParInt(NumlevelsName);
}

double
Parameters::evalPyramidscale(const OP_Inputs* input)
{
	return input->getParDouble(PyramidscaleName);
}

bool
Parameters::evalFastpyramids(const OP_Inputs* input)
{
	return input->getParInt(FastpyramidsName) ? true : false;
}

int
Parameters::evalWindowsize(const OP_Inputs* input)
{
	return input->getParInt(WindowsizeName);
}

int
Parameters::evalIterations(const OP_Inputs* input)
{
	return input->getParInt(IterationsName);
}

int
Parameters::evalPolyn(const OP_Inputs* input)
{
	return input->getParInt(PolynName);
}

double
Parameters::evalPolysigma(const OP_Inputs* input)
{
	return input->getParDouble(PolysigmaName);
}

bool
Parameters::evalUsegaussianfilter(const OP_Inputs* input)
{
	return input->getParInt(UsegaussianfilterName) ? true : false;
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(OP_ParameterManager* manager)
{
	{
		OP_NumericParameter p;
		p.name = NumlevelsName;
		p.label = NumlevelsLabel;
		p.page = "Optical Flow";
		p.defaultValues[0] = 5;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendInt(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = PyramidscaleName;
		p.label = PyramidscaleLabel;
		p.page = "Optical Flow";
		p.defaultValues[0] = 0.5;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 1.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = true;
		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = FastpyramidsName;
		p.label = FastpyramidsLabel;
		p.page = "Optical Flow";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = WindowsizeName;
		p.label = WindowsizeLabel;
		p.page = "Optical Flow";
		p.defaultValues[0] = 13;
		p.minSliders[0] = 1.0;
		p.maxSliders[0] = 100.0;
		p.minValues[0] = 1.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendInt(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = IterationsName;
		p.label = IterationsLabel;
		p.page = "Optical Flow";
		p.defaultValues[0] = 10;
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
		p.name = PolynName;
		p.label = PolynLabel;
		p.page = "Optical Flow";
		p.defaultValues[0] = 5;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendInt(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = PolysigmaName;
		p.label = PolysigmaLabel;
		p.page = "Optical Flow";
		p.defaultValues[0] = 1.1;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 2.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = UsegaussianfilterName;
		p.label = UsegaussianfilterLabel;
		p.page = "Optical Flow";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}


}

#pragma endregion