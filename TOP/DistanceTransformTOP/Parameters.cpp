#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

DistancetypeMenuItems
Parameters::evalDistancetype(const TD::OP_Inputs* inputs)
{
	return static_cast<DistancetypeMenuItems>(inputs->getParInt(DistancetypeName));
}

MasksizeMenuItems
Parameters::evalMasksize(const TD::OP_Inputs* inputs)
{
	return static_cast<MasksizeMenuItems>(inputs->getParInt(MasksizeName));
}

bool
Parameters::evalNormalize(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(NormalizeName) ? true : false;
}

ChannelMenuItems
Parameters::evalChannel(const TD::OP_Inputs* inputs)
{
	return static_cast<ChannelMenuItems>(inputs->getParInt(ChannelName));
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(TD::OP_ParameterManager* manager)
{
	{
		TD::OP_StringParameter p;
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
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_StringParameter p;
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
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = NormalizeName;
		p.label = NormalizeLabel;
		p.page = "Transform";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_StringParameter p;
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
		TD::OP_ParAppendResult res = manager->appendMenu(p, Names.size(), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}


}

#pragma endregion