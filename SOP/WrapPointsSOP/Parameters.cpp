#include "Parameters.h"
#include "CPlusPlus_Common.h"

bool
Parameters::evalParms(const OP_Inputs* input)
{
	bool changed = false;
	Rays	tmprays = static_cast<Rays>(input->getParInt(RAYS_NAME));
	changed |= tmprays != rays;
	rays = tmprays;

	bool needDirection = rays == Rays::Parallel;
	input->enablePar(DIRECTION_NAME, needDirection);
	if (needDirection)
	{
		double	tmpdirection[3];
		input->getParDouble3(DIRECTION_NAME, tmpdirection[0], tmpdirection[1], tmpdirection[2]);
		changed |= memcmp(direction, tmpdirection, sizeof(direction)) != 0;
		memcpy(direction, tmpdirection, sizeof(direction));
	}

	bool needOrigin = rays == Rays::Radial;
	input->enablePar(ORIGIN_NAME, needOrigin);
	if (needOrigin)
	{
		double	tmporigin[3];
		input->getParDouble3(ORIGIN_NAME, tmporigin[0], tmporigin[1], tmporigin[2]);
		changed |= memcmp(origin, tmporigin, sizeof(origin)) != 0;
		memcpy(origin, tmporigin, sizeof(origin));
	}

	bool	tmpreverse = input->getParInt(REVERSE_NAME) ? true : false;
	changed |= tmpreverse != reverse;
	reverse = tmpreverse;

	Color	tmphitcolor;
	tmphitcolor.r = static_cast<float>(input->getParDouble(HITCOLOR_NAME, 0));
	tmphitcolor.g = static_cast<float>(input->getParDouble(HITCOLOR_NAME, 1));
	tmphitcolor.b = static_cast<float>(input->getParDouble(HITCOLOR_NAME, 2));
	tmphitcolor.a = static_cast<float>(input->getParDouble(HITCOLOR_NAME, 3));
	changed |= memcmp(&hitcolor, &tmphitcolor, sizeof(hitcolor)) != 0;
	hitcolor = tmphitcolor;

	Color	tmpmisscolor;
	tmpmisscolor.r = static_cast<float>(input->getParDouble(MISSCOLOR_NAME, 0));
	tmpmisscolor.g = static_cast<float>(input->getParDouble(MISSCOLOR_NAME, 1));
	tmpmisscolor.b = static_cast<float>(input->getParDouble(MISSCOLOR_NAME, 2));
	tmpmisscolor.a = static_cast<float>(input->getParDouble(MISSCOLOR_NAME, 3));
	changed |= memcmp(&misscolor, &tmpmisscolor, sizeof(misscolor)) != 0;
	misscolor = tmpmisscolor;

	double	tmpscale = input->getParDouble(SCALE_NAME);
	changed |= tmpscale != scale;
	scale = tmpscale;

	return changed;
}

void
Parameters::setupParms(OP_ParameterManager* manager)
{

	{
		OP_StringParameter p;
		p.name = RAYS_NAME;
		p.label = "Rays";
		p.page = "Cast Ray";
		p.defaultValue = "Parallel";

		const char*	names[] = { "Parallel", "Radial" };
		const char*	labels[] = { "Parallel", "Radial" };
		OP_ParAppendResult res = manager->appendMenu(p, 2, names, labels);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = DIRECTION_NAME;
		p.label = "Direction";
		p.page = "Cast Ray";

		const double defaultValues[] = { 0.0, 0.0, 0.0 };
		const double minSliders[] = { 0.0, 0.0, 0.0 };
		const double maxSliders[] = { 1.0, 1.0, 1.0 };
		const double minValues[] = { 0.0, 0.0, 0.0 };
		const double maxValues[] = { 1.0, 1.0, 1.0 };
		const bool clampMins[] = { false, false, false };
		const bool clampMaxes[] = { false, false, false };
		for (int i = 0; i < 3; ++i)
		{
			p.defaultValues[i] = defaultValues[i];
			p.minSliders[i] = minSliders[i];
			p.maxSliders[i] = maxSliders[i];
			p.minValues[i] = minValues[i];
			p.maxValues[i] = maxValues[i];
			p.clampMins[i] = clampMins[i];
			p.clampMaxes[i] = clampMaxes[i];
		}
		OP_ParAppendResult res = manager->appendXYZ(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = ORIGIN_NAME;
		p.label = "Destination";
		p.page = "Cast Ray";

		const double defaultValues[] = { 0.0, 0.0, 0.0 };
		const double minSliders[] = { 0.0, 0.0, 0.0 };
		const double maxSliders[] = { 1.0, 1.0, 1.0 };
		const double minValues[] = { 0.0, 0.0, 0.0 };
		const double maxValues[] = { 1.0, 1.0, 1.0 };
		const bool clampMins[] = { false, false, false };
		const bool clampMaxes[] = { false, false, false };
		for (int i = 0; i < 3; ++i)
		{
			p.defaultValues[i] = defaultValues[i];
			p.minSliders[i] = minSliders[i];
			p.maxSliders[i] = maxSliders[i];
			p.minValues[i] = minValues[i];
			p.maxValues[i] = maxValues[i];
			p.clampMins[i] = clampMins[i];
			p.clampMaxes[i] = clampMaxes[i];
		}
		OP_ParAppendResult res = manager->appendXYZ(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = REVERSE_NAME;
		p.label = "Reverse";
		p.page = "Cast Ray";

		p.defaultValues[0] = false;
		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = HITCOLOR_NAME;
		p.label = "Hit Color";
		p.page = "Cast Ray";

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
		p.name = MISSCOLOR_NAME;
		p.label = "Miss Color";
		p.page = "Cast Ray";

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

	{
		OP_NumericParameter p;
		p.name = SCALE_NAME;
		p.label = "Scale";
		p.page = "Cast Ray";

		p.defaultValues[0] = 1.0;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 1.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = false;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendFloat(p, 1);

		assert(res == OP_ParAppendResult::Success);
	}
}
