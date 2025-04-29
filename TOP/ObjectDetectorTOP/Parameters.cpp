#include <string>
#include <array>
#include "CPlusPlus_Common.h"
#include "Parameters.h"

#pragma region Evals

std::string
Parameters::evalClassifier(const TD::OP_Inputs* inputs)
{
	return inputs->getParFilePath(ClassifierName);
}

double
Parameters::evalScalefactor(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(ScalefactorName);
}

int
Parameters::evalMinneighbors(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(MinneighborsName);
}

bool
Parameters::evalLimitobjectsize(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(LimitobjectsizeName) ? true : false;
}

double
Parameters::evalMinobjectwidth(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(MinobjectwidthName);
}

double
Parameters::evalMinobjectheight(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(MinobjectheightName);
}

double
Parameters::evalMaxobjectwidth(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(MaxobjectwidthName);
}

double
Parameters::evalMaxobjectheight(const TD::OP_Inputs* inputs)
{
	return inputs->getParDouble(MaxobjectheightName);
}

bool
Parameters::evalDrawboundingbox(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(DrawboundingboxName) ? true : false;
}

bool
Parameters::evalLimitobjectsdetected(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(LimitobjectsdetectedName) ? true : false;
}

int
Parameters::evalMaximumobjects(const TD::OP_Inputs* inputs)
{
	return inputs->getParInt(MaximumobjectsName);
}


#pragma endregion

#pragma region Setup

void
Parameters::setup(TD::OP_ParameterManager* manager)
{
	{
		TD::OP_StringParameter p;
		p.name = ClassifierName;
		p.label = ClassifierLabel;
		p.page = "Object Detector";
		p.defaultValue = "";
		TD::OP_ParAppendResult res = manager->appendFile(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
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
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
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
		TD::OP_ParAppendResult res = manager->appendInt(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = LimitobjectsizeName;
		p.label = LimitobjectsizeLabel;
		p.page = "Object Detector";
		p.defaultValues[0] = true;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
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
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
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
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
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
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
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
		TD::OP_ParAppendResult res = manager->appendFloat(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = DrawboundingboxName;
		p.label = DrawboundingboxLabel;
		p.page = "Object Detector";
		p.defaultValues[0] = true;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
		p.name = LimitobjectsdetectedName;
		p.label = LimitobjectsdetectedLabel;
		p.page = "Object Detector";
		p.defaultValues[0] = false;

		TD::OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}

	{
		TD::OP_NumericParameter p;
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
		TD::OP_ParAppendResult res = manager->appendInt(p);

		assert(res == TD::OP_ParAppendResult::Success);
	}


}

#pragma endregion