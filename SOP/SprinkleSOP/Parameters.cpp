#include "Parameters.h"
#include "CPlusPlus_Common.h"

void
Parameters::evalParms(const OP_Inputs* input)
{
	double tmpseed = input->getParDouble(SEED_NAME);
	Generate tmpgenerate = static_cast<Generate>(input->getParInt(GENERATE_NAME));
	int tmppointcount = input->getParInt(POINTCOUNT_NAME);
	bool tmpforcedistance = input->getParInt(FORCEDISTANCE_NAME) ? true : false;
	double tmppointdistance = input->getParDouble(POINTDISTANCE_NAME);

	changed = tmpseed != seed || tmpgenerate != generate ||
		tmppointcount != pointcount || tmpforcedistance != forcedistance || tmppointdistance != pointdistance;

	seed = tmpseed;
	generate = tmpgenerate;
	pointcount = tmppointcount;
	forcedistance = tmpforcedistance;
	pointdistance = tmppointdistance;

	input->enablePar(POINTDISTANCE_NAME, forcedistance);
}

void
Parameters::setupParms(OP_ParameterManager* manager)
{
	{
		OP_NumericParameter p;
		p.name = SEED_NAME;
		p.label = "Seed";
		p.page = "Sprinkle";

		p.defaultValues[0] = 1.0;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 10.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendFloat(p, 1);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter p;
		p.name = GENERATE_NAME;
		p.label = "Generate";
		p.page = "Sprinkle";
		p.defaultValue = "Density";

		const char*	names[] = { "Area", "Primitive", "Boundingbox", "Volume" };
		const char*	labels[] = { "Surface Area", "Per Primitive", "Bounding Box", "Inside Volume" };
		OP_ParAppendResult res = manager->appendMenu(p, 4, names, labels);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = POINTCOUNT_NAME;
		p.label = "Point Count";
		p.page = "Sprinkle";

		p.defaultValues[0] = 100;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 1000.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendInt(p, 1);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = FORCEDISTANCE_NAME;
		p.label = "Separate Points";
		p.page = "Sprinkle";

		p.defaultValues[0] = false;
		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = POINTDISTANCE_NAME;
		p.label = "Minimum Distance";
		p.page = "Sprinkle";

		p.defaultValues[0] = 0.01;
		p.minSliders[0] = 0.0;
		p.maxSliders[0] = 1.0;
		p.minValues[0] = 0.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendFloat(p, 1);

		assert(res == OP_ParAppendResult::Success);
	}


}
