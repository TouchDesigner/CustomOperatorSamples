#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

int
Parameters::evalBitspercolor(const OP_Inputs* input)
{
	return input->getParInt(BitspercolorName);
}

bool
Parameters::evalDither(const OP_Inputs* input)
{
	return input->getParInt(DitherName) ? true : false;
}

bool
Parameters::evalMultithreaded(const OP_Inputs* input)
{
	return input->getParInt(MultithreadedName) ? true : false;
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
		p.name = BitspercolorName;
		p.label = BitspercolorLabel;
		p.page = "Filter";
		p.defaultValues[0] = 2;
		p.minSliders[0] = 1.0;
		p.maxSliders[0] = 8.0;
		p.minValues[0] = 1.0;
		p.maxValues[0] = 8.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = true;
		OP_ParAppendResult res = manager->appendInt(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = DitherName;
		p.label = DitherLabel;
		p.page = "Filter";
		p.defaultValues[0] = true;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = MultithreadedName;
		p.label = MultithreadedLabel;
		p.page = "Filter";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter p;
		p.name = DownloadtypeName;
		p.label = DownloadtypeLabel;
		p.page = "Filter";
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