#include "Parameters.h"
#include "CPlusPlus_Common.h"

bool
Parameters::evalParms(const OP_Inputs* input)
{
	bool changed = false;
	double	tmpinsidevalue[4];
	input->getParDouble4(INSIDEVALUE_NAME, tmpinsidevalue[0], tmpinsidevalue[1], tmpinsidevalue[2], tmpinsidevalue[3]);
	changed |= memcmp(insidevalue, tmpinsidevalue, sizeof(insidevalue)) != 0;
	memcpy(insidevalue, tmpinsidevalue, sizeof(insidevalue));

	double	tmpoutsidevalue[4];
	input->getParDouble4(OUTSIDEVALUE_NAME, tmpoutsidevalue[0], tmpoutsidevalue[1], tmpoutsidevalue[2], tmpoutsidevalue[3]);
	changed |= memcmp(outsidevalue, tmpoutsidevalue, sizeof(outsidevalue)) != 0;
	memcpy(outsidevalue, tmpoutsidevalue, sizeof(outsidevalue));

	return changed;
}

void
Parameters::setupParms(OP_ParameterManager* manager)
{
	{
		OP_NumericParameter p;
		p.name = INSIDEVALUE_NAME;
		p.label = "Inside Color";
		p.page = "Intersection";

		const double defaultValues[] = { 1.0, 1.0, 1.0, 1.0 };
		const double minSliders[] = { 0.0, 0.0, 0.0, 0.0 };
		const double maxSliders[] = { 1.0, 1.0, 1.0, 1.0 };
		const double minValues[] = { 0.0, 0.0, 0.0, 0.0 };
		const double maxValues[] = { 1.0, 1.0, 1.0, 1.0 };
		const bool clampMins[] = { false, false, false, false };
		const bool clampMaxes[] = { false, false, false, false };
		for (int i = 0; i < 4; ++i)
		{
			p.defaultValues[i] = defaultValues[i];
			p.minSliders[i] = minSliders[i];
			p.maxSliders[i] = maxSliders[i];
			p.minValues[i] = minValues[i];
			p.maxValues[i] = maxValues[i];
			p.clampMins[i] = clampMins[i];
			p.clampMaxes[i] = clampMaxes[i];
		}
		OP_ParAppendResult res = manager->appendRGBA(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = OUTSIDEVALUE_NAME;
		p.label = "Outside Color";
		p.page = "Intersection";

		const double defaultValues[] = { 0.0, 0.0, 0.0, 1.0 };
		const double minSliders[] = { 0.0, 0.0, 0.0, 0.0 };
		const double maxSliders[] = { 1.0, 1.0, 1.0, 1.0 };
		const double minValues[] = { 0.0, 0.0, 0.0, 0.0 };
		const double maxValues[] = { 1.0, 1.0, 1.0, 1.0 };
		const bool clampMins[] = { false, false, false, false };
		const bool clampMaxes[] = { false, false, false, false };
		for (int i = 0; i < 4; ++i)
		{
			p.defaultValues[i] = defaultValues[i];
			p.minSliders[i] = minSliders[i];
			p.maxSliders[i] = maxSliders[i];
			p.minValues[i] = minValues[i];
			p.maxValues[i] = maxValues[i];
			p.clampMins[i] = clampMins[i];
			p.clampMaxes[i] = clampMaxes[i];
		}
		OP_ParAppendResult res = manager->appendRGBA(p);

		assert(res == OP_ParAppendResult::Success);
	}
}
