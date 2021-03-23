#include "Parameters.h"
#include "CPlusPlus_Common.h"

bool
Parameters::evalParms(const OP_Inputs* input)
{
	bool changed = false;
	Distancetype	tmpdistancetype = static_cast<Distancetype>(input->getParInt(DISTANCETYPE_NAME));
	changed |= tmpdistancetype != distancetype;
	distancetype = tmpdistancetype;

	bool needMask = distancetype == Distancetype::L2;
	input->enablePar(MASKSIZE_NAME, needMask);
	if (needMask)
	{
		Masksize	tmpmasksize = static_cast<Masksize>(input->getParInt(MASKSIZE_NAME));
		changed |= tmpmasksize != masksize;
		masksize = tmpmasksize;
	}

	bool	tmpnormalize = input->getParInt(NORMALIZE_NAME) ? true : false;
	changed |= tmpnormalize != normalize;
	normalize = tmpnormalize;

	OP_TOPInputDownloadType tmpdownloadtype = static_cast<OP_TOPInputDownloadType>(input->getParInt(DOWNLOADTYPE_NAME));
	changed |= tmpdownloadtype != downloadtype;
	downloadtype = tmpdownloadtype;
        
        Channel tmpchannel = static_cast<Channel>(input->getParInt(CHANNEL_NAME));
        changed |= tmpchannel != channel;
        channel = tmpchannel;

	return changed;
}

void
Parameters::setupParms(OP_ParameterManager* manager)
{

	{
		OP_StringParameter p;
		p.name = DISTANCETYPE_NAME;
		p.label = "Distance Type";
		p.page = "Transform";
		p.defaultValue = "User";

		const char*	names[] = { "L1", "L2", "C" };
		const char*	labels[] = { "L1", "L2", "C" };
		OP_ParAppendResult res = manager->appendMenu(p, 3, names, labels);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter p;
		p.name = MASKSIZE_NAME;
		p.label = "Mask Size";
		p.page = "Transform";
		p.defaultValue = "3x3";

		const char*	names[] = { "Three", "Five", "Precise" };
		const char*	labels[] = { "3x3", "5x5", "Precise" };
		OP_ParAppendResult res = manager->appendMenu(p, 3, names, labels);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = NORMALIZE_NAME;
		p.label = "Normalize";
		p.page = "Transform";

		p.defaultValues[0] = false;
		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter sp;
		sp.name = DOWNLOADTYPE_NAME;
		sp.label = "Download Type";
		sp.page = "Transform";

		const char* names[] = { "Delayed", "Instant" };
		const char* labels[] = { "Delayed", "Instant" };

		OP_ParAppendResult res = manager->appendMenu(sp, 2, names, labels);
		assert(res == OP_ParAppendResult::Success);
	}

        {
		OP_StringParameter p;
		p.name = CHANNEL_NAME;
		p.label = "Channel";
		p.page = "Transform";
		p.defaultValue = "R";

		const char*	names[] = { "R", "G", "B", "A" };
		const char*	labels[] = { "R", "G", "B", "A" };
		OP_ParAppendResult res = manager->appendMenu(p, 4, names, labels);

		assert(res == OP_ParAppendResult::Success);
	}
}
