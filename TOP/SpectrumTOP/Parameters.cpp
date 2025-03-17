#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

ModeMenuItems
Parameters::evalMode(const TD::OP_Inputs* inputs)
{
	return static_cast<ModeMenuItems>(inputs->getParInt(ModeName));
}

CoordMenuItems
Parameters::evalCoord(const TD::OP_Inputs* inputs)
{
	return static_cast<CoordMenuItems>(inputs->getParInt(CoordName));
}

ChanMenuItems
Parameters::evalChan(const TD::OP_Inputs* inputs)
{
	return static_cast<ChanMenuItems>(inputs->getParInt(ChanName));
}

bool
Parameters::evalTransrows(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(TransrowsName) ? true : false;
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(TD::OP_ParameterManager* manager)
{
	{
		TD::OP_StringParameter p;
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
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_StringParameter p;
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
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_StringParameter p;
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
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = TransrowsName;
		p.label = TransrowsLabel;
		p.page = "Spectrum";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}


}

#pragma endregion