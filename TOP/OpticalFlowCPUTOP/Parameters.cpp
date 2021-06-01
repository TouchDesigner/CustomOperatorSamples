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

bool
Parameters::evalUsepreviousflow(const OP_Inputs* input)
{
	return input->getParInt(UsepreviousflowName) ? true : false;
}

ChannelMenuItems
Parameters::evalChannel(const OP_Inputs* input)
{
	return static_cast<ChannelMenuItems>(input->getParInt(ChannelName));
}

DownloadtypeMenuItems
Parameters::evalDownloadtype(const OP_Inputs* input)
{
	return static_cast<DownloadtypeMenuItems>(input->getParInt(DownloadtypeName));
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
		p.maxSliders[0] = 0.5;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 0.5;
		p.clampMins[0] = true;
		p.clampMaxes[0] = true;
		OP_ParAppendResult res = manager->appendFloat(p);

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
		p.maxSliders[0] = 50.0;
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
		p.minSliders[0] = 5.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 5.0;
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

	{
		OP_NumericParameter p;
		p.name = UsepreviousflowName;
		p.label = UsepreviousflowLabel;
		p.page = "Optical Flow";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter p;
		p.name = ChannelName;
		p.label = ChannelLabel;
		p.page = "Optical Flow";
		p.defaultValue = "R";
		std::array<const char*, 4> Names =
		{
			"R",
			"G",
			"B",
			"A"
		};
		std::array<const char*, 4> Labels =
		{
			"R",
			"G",
			"B",
			"A"
		};
		OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter p;
		p.name = DownloadtypeName;
		p.label = DownloadtypeLabel;
		p.page = "Optical Flow";
		p.defaultValue = "Delayed";
		std::array<const char*, 2> Names =
		{
			"Delayed",
			"Instant"
		};
		std::array<const char*, 2> Labels =
		{
			"Delayed",
			"Instant"
		};
		OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == OP_ParAppendResult::Success);
	}


}

#pragma endregion