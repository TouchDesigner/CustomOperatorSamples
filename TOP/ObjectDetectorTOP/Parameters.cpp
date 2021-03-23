#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

std::string
Parameters::evalClassifier(const OP_Inputs* input)
{
	return input->getParFilePath(ClassifierName);
}

double
Parameters::evalScalefactor(const OP_Inputs* input)
{
	return input->getParDouble(ScalefactorName);
}

int
Parameters::evalMinneighbors(const OP_Inputs* input)
{
	return input->getParInt(MinneighborsName);
}

bool
Parameters::evalLimitobjectsize(const OP_Inputs* input)
{
	return input->getParInt(LimitobjectsizeName) ? true : false;
}

double
Parameters::evalMinobjectwidth(const OP_Inputs* input)
{
	return input->getParDouble(MinobjectwidthName);
}

double
Parameters::evalMinobjectheight(const OP_Inputs* input)
{
	return input->getParDouble(MinobjectheightName);
}

double
Parameters::evalMaxobjectwidth(const OP_Inputs* input)
{
	return input->getParDouble(MaxobjectwidthName);
}

double
Parameters::evalMaxobjectheight(const OP_Inputs* input)
{
	return input->getParDouble(MaxobjectheightName);
}

bool
Parameters::evalDrawboundingbox(const OP_Inputs* input)
{
	return input->getParInt(DrawboundingboxName) ? true : false;
}

bool
Parameters::evalLimitobjectsdetected(const OP_Inputs* input)
{
	return input->getParInt(LimitobjectsdetectedName) ? true : false;
}

int
Parameters::evalMaximumobjects(const OP_Inputs* input)
{
	return input->getParInt(MaximumobjectsName);
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
		OP_StringParameter p;
		p.name = ClassifierName;
		p.label = ClassifierLabel;
		p.page = "Object Detector";
		p.defaultValue = "";
		OP_ParAppendResult res = manager->appendFile(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = ScalefactorName;
		p.label = ScalefactorLabel;
		p.page = "Object Detector";
		p.defaultValues[0] = 1.05;
		p.minSliders[0] = 1.0;
		p.maxSliders[0] = 5.0;
		p.minValues[0] = 1.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = MinneighborsName;
		p.label = MinneighborsLabel;
		p.page = "Object Detector";
		p.defaultValues[0] = 3;
		p.minSliders[0] = 1.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 1.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendInt(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = LimitobjectsizeName;
		p.label = LimitobjectsizeLabel;
		p.page = "Object Detector";
		p.defaultValues[0] = true;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = MinobjectwidthName;
		p.label = MinobjectwidthLabel;
		p.page = "Object Detector";
		p.defaultValues[0] = 0.1;
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
		p.name = MinobjectheightName;
		p.label = MinobjectheightLabel;
		p.page = "Object Detector";
		p.defaultValues[0] = 0.1;
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
		p.name = MaxobjectwidthName;
		p.label = MaxobjectwidthLabel;
		p.page = "Object Detector";
		p.defaultValues[0] = 0.9;
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
		p.name = MaxobjectheightName;
		p.label = MaxobjectheightLabel;
		p.page = "Object Detector";
		p.defaultValues[0] = 0.9;
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
		p.name = DrawboundingboxName;
		p.label = DrawboundingboxLabel;
		p.page = "Object Detector";
		p.defaultValues[0] = true;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = LimitobjectsdetectedName;
		p.label = LimitobjectsdetectedLabel;
		p.page = "Object Detector";
		p.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = MaximumobjectsName;
		p.label = MaximumobjectsLabel;
		p.page = "Object Detector";
		p.defaultValues[0] = 10;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 20.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendInt(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter p;
		p.name = DownloadtypeName;
		p.label = DownloadtypeLabel;
		p.page = "Object Detector";
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