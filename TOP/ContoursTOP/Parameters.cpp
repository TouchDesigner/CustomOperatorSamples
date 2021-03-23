#include "Parameters.h"
#include "CPlusPlus_Common.h"

bool
Parameters::evalParms(const OP_Inputs* input)
{
	bool changed = false;
	Mode	tmpmode = static_cast<Mode>(input->getParInt(MODE_NAME));
	changed |= tmpmode != mode;
	mode = tmpmode;

	Method	tmpmethod = static_cast<Method>(input->getParInt(METHOD_NAME));
	changed |= tmpmethod != method;
	method = tmpmethod;

        if (input->getNumInputs() == 2)
        {
                bool	tmpapplywatershed = input->getParInt(APPLYWATERSHED_NAME) ? true : false;
                changed |= tmpapplywatershed != applywatershed;
                applywatershed = tmpapplywatershed;
                input->enablePar(APPLYWATERSHED_NAME, true);
        }
        else 
        {
                input->enablePar(APPLYWATERSHED_NAME, false);
        }

	bool	tmpselectobject = input->getParInt(SELECTOBJECT_NAME) ? true : false;
	changed |= tmpselectobject != selectobject;
	selectobject = tmpselectobject;

	input->enablePar(OBJECT_NAME, selectobject);
	if (selectobject)
	{
		int	tmpobject = input->getParInt(OBJECT_NAME);
		changed |= tmpobject != object;
		object = tmpobject;
	}

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
		p.name = MODE_NAME;
		p.label = "Mode";
		p.page = "Contours";
		p.defaultValue = "External";

		const char*	names[] = { "External", "List", "Ccomp", "Tree" };
		const char*	labels[] = { "External", "List", "CComp", "Tree" };
		OP_ParAppendResult res = manager->appendMenu(p, 4, names, labels);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter p;
		p.name = METHOD_NAME;
		p.label = "Method";
		p.page = "Contours";
		p.defaultValue = "None";

		const char*	names[] = { "None", "Simple", "Tcl1", "Tckcos" };
		const char*	labels[] = { "None", "Simple", "Teh-Chin L1", "Teh-Chin KCos" };
		OP_ParAppendResult res = manager->appendMenu(p, 4, names, labels);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = APPLYWATERSHED_NAME;
		p.label = "Watershed";
		p.page = "Contours";

		p.defaultValues[0] = false;
		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = SELECTOBJECT_NAME;
		p.label = "Select Object";
		p.page = "Contours";

		p.defaultValues[0] = false;
		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = OBJECT_NAME;
		p.label = "Object";
		p.page = "Contours";

		p.defaultValues[0] = 0;
		p.minSliders[0] = -1.0;
		p.maxSliders[0] = 100.0;
		p.minValues[0] = -1.0;
		p.maxValues[0] = 1.0;
		p.clampMins[0] = true;
		p.clampMaxes[0] = false;
		OP_ParAppendResult res = manager->appendInt(p, 1);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter sp;
		sp.name = DOWNLOADTYPE_NAME;
		sp.label = "Download Type";
		sp.page = "Contours";

		const char* names[] = { "Delayed", "Instant" };
		const char* labels[] = { "Delayed", "Instant" };

		OP_ParAppendResult res = manager->appendMenu(sp, 2, names, labels);
		assert(res == OP_ParAppendResult::Success);
	}

        {
		OP_StringParameter p;
		p.name = CHANNEL_NAME;
		p.label = "Channel";
		p.page = "Contours";
		p.defaultValue = "R";

		const char*	names[] = { "R", "G", "B", "A" };
		const char*	labels[] = { "R", "G", "B", "A" };
		OP_ParAppendResult res = manager->appendMenu(p, 4, names, labels);

		assert(res == OP_ParAppendResult::Success);
	}
}
