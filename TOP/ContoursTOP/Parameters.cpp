#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

ModeMenuItems
Parameters::evalMode(const TD::OP_Inputs* input)
{
	return static_cast<ModeMenuItems>(input->getParInt(ModeName));
}

MethodMenuItems
Parameters::evalMethod(const TD::OP_Inputs* input)
{
	return static_cast<MethodMenuItems>(input->getParInt(MethodName));
}

bool
Parameters::evalWatershed(const TD::OP_Inputs* input)
{
	return input->getParInt(WatershedName) ? true : false;
}

bool
Parameters::evalSelectobject(const TD::OP_Inputs* input)
{
	return input->getParInt(SelectobjectName) ? true : false;
}

int
Parameters::evalObject(const TD::OP_Inputs* input)
{
	return input->getParInt(ObjectName);
}

DownloadtypeMenuItems
Parameters::evalDownloadtype(const TD::OP_Inputs* input)
{
	return static_cast<DownloadtypeMenuItems>(input->getParInt(DownloadtypeName));
}

ChannelMenuItems
Parameters::evalChannel(const TD::OP_Inputs* input)
{
	return static_cast<ChannelMenuItems>(input->getParInt(ChannelName));
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
		p.page = "Contours";
		p.defaultValue = "External";
		std::array<const char*, 4> Names =
		{
			"External",
			"List",
			"Ccomp",
			"Tree"
		};
		std::array<const char*, 4> Labels =
		{
			"External",
			"List",
			"Ccomp",
			"Tree"
		};
		TD::OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_StringParameter p;
		p.name = MethodName;
		p.label = MethodLabel;
		p.page = "Contours";
		p.defaultValue = "None";
		std::array<const char*, 4> Names =
		{
			"None",
			"Simple",
			"Tcl1",
			"Tckcos"
		};
		std::array<const char*, 4> Labels =
		{
			"None",
			"Simple",
			"Tcl1",
			"Tckcos"
		};
		TD::OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = WatershedName;
		p.label = WatershedLabel;
		p.page = "Contours";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = SelectobjectName;
		p.label = SelectobjectLabel;
		p.page = "Contours";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = ObjectName;
		p.label = ObjectLabel;
		p.page = "Contours";
		p.defaultValues[0] = 0;
		p.minSliders[0] = -1.0;
		p.maxSliders[0] = 100.0;
		p.minValues[0] = -1.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		TD::OP_ParAppendResult res = manager->appendInt(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_StringParameter p;
		p.name = DownloadtypeName;
		p.label = DownloadtypeLabel;
		p.page = "Contours";
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
		TD::OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_StringParameter p;
		p.name = ChannelName;
		p.label = ChannelLabel;
		p.page = "Contours";
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
		TD::OP_ParAppendResult res = manager->appendMenu(p, int(Names.size()), Names.data(), Labels.data());

		assert(res == TD::OP_ParAppendResult::Success);
	}


}

#pragma endregion