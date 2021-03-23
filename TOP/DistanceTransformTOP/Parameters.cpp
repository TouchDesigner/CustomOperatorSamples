#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

DistancetypeMenuItems
Parameters::evalDistancetype(const OP_Inputs* input)
{
	return static_cast<DistancetypeMenuItems>(input->getParInt(DistancetypeName));
}

MasksizeMenuItems
Parameters::evalMasksize(const OP_Inputs* input)
{
	return static_cast<MasksizeMenuItems>(input->getParInt(MasksizeName));
}

bool
Parameters::evalNormalize(const OP_Inputs* input)
{
	return input->getParInt(NormalizeName) ? true : false;
}

DownloadtypeMenuItems
Parameters::evalDownloadtype(const OP_Inputs* input)
{
	return static_cast<DownloadtypeMenuItems>(input->getParInt(DownloadtypeName));
}

ChannelMenuItems
Parameters::evalChannel(const OP_Inputs* input)
{
	return static_cast<ChannelMenuItems>(input->getParInt(ChannelName));
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(OP_ParameterManager* manager)
{
	{
		OP_StringParameter p;
		p.name = DistancetypeName;
		p.label = DistancetypeLabel;
		p.page = "Transform";
		p.defaultValue = "L1";
		std::array<const char*, 3> Names =
		{
			"L1",
			"L2",
			"C"
		};
		std::array<const char*, 3> Labels =
		{
			"L1",
			"L2",
			"C"
		};
		OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter p;
		p.name = MasksizeName;
		p.label = MasksizeLabel;
		p.page = "Transform";
		p.defaultValue = "Three";
		std::array<const char*, 3> Names =
		{
			"Three",
			"Five",
			"Precise"
		};
		std::array<const char*, 3> Labels =
		{
			"3x3",
			"5x5",
			"Precise"
		};
		OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = NormalizeName;
		p.label = NormalizeLabel;
		p.page = "Transform";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter p;
		p.name = DownloadtypeName;
		p.label = DownloadtypeLabel;
		p.page = "Transform";
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

	{
		OP_StringParameter p;
		p.name = ChannelName;
		p.label = ChannelLabel;
		p.page = "Transform";
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


}

#pragma endregion