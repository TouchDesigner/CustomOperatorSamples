#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

ModeMenuItems
Parameters::evalMode(const OP_Inputs* input)
{
	return static_cast<ModeMenuItems>(input->getParInt(ModeName));
}

CoordMenuItems
Parameters::evalCoord(const OP_Inputs* input)
{
	return static_cast<CoordMenuItems>(input->getParInt(CoordName));
}

ChanMenuItems
Parameters::evalChan(const OP_Inputs* input)
{
	return static_cast<ChanMenuItems>(input->getParInt(ChanName));
}

bool
Parameters::evalTransrows(const OP_Inputs* input)
{
	return input->getParInt(TransrowsName) ? true : false;
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(OP_ParameterManager* manager)
{
	{
		OP_StringParameter p;
		p.name = ModeName;
		p.label = ModeLabel;
		p.page = "Spectrum";
		p.defaultValue = "dft";
		std::array<const char*, 2> Names =
		{
			"dft",
			"idft"
		};
		std::array<const char*, 2> Labels =
		{
			"Discrete Fourier Transform",
			"Inverse Discrete Fourier Transform"
		};
		OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter p;
		p.name = CoordName;
		p.label = CoordLabel;
		p.page = "Spectrum";
		p.defaultValue = "polar";
		std::array<const char*, 2> Names =
		{
			"polar",
			"cartesian"
		};
		std::array<const char*, 2> Labels =
		{
			"Polar (Magnitude, Phase)",
			"Cartesian (Real, Imaginary)"
		};
		OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter p;
		p.name = ChanName;
		p.label = ChanLabel;
		p.page = "Spectrum";
		p.defaultValue = "r";
		std::array<const char*, 4> Names =
		{
			"r",
			"g",
			"b",
			"a"
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
		OP_NumericParameter p;
		p.name = TransrowsName;
		p.label = TransrowsLabel;
		p.page = "Spectrum";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}


}

#pragma endregion