#include "Parameters.h"
#include "CPlusPlus_Common.h"

bool
Parameters::evalParms(const OP_Inputs* input)
{
	bool changed = false;
	double	tmplowthreshold = input->getParDouble(LOWTHRESHOLD_NAME);
	changed |= tmplowthreshold != lowthreshold;
	lowthreshold = tmplowthreshold;

	double	tmphighthreshold = input->getParDouble(HIGHTHRESHOLD_NAME);
	changed |= tmphighthreshold != highthreshold;
	highthreshold = tmphighthreshold;

	int	tmpapperturesize = input->getParInt(APPERTURESIZE_NAME);
	changed |= tmpapperturesize != apperturesize;
	apperturesize = tmpapperturesize;

	bool	tmpl2gradient = input->getParInt(L2GRADIENT_NAME) ? true : false;
	changed |= tmpl2gradient != l2gradient;
	l2gradient = tmpl2gradient;

	return changed;
}

void
Parameters::setupParms(OP_ParameterManager* manager)
{

	{
		OP_NumericParameter p;
		p.name = LOWTHRESHOLD_NAME;
		p.label = "Low Threshold";
		p.page = "Edge Detector";

		p.defaultValues[0] = 0.1;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 1.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = true;
		OP_ParAppendResult res = manager->appendFloat(p, 1);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = HIGHTHRESHOLD_NAME;
		p.label = "High Threshold";
		p.page = "Edge Detector";

		p.defaultValues[0] = 0.9;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 1.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = true;
		OP_ParAppendResult res = manager->appendFloat(p, 1);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = APPERTURESIZE_NAME;
		p.label = "Apperture Size";
		p.page = "Edge Detector";

		p.defaultValues[0] = 3;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendInt(p, 1);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = L2GRADIENT_NAME;
		p.label = "L2 Gradient";
		p.page = "Edge Detector";

		p.defaultValues[0] = false;
		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}
}
