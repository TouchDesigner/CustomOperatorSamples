#include "Parameters.h"
#include "CPlusPlus_Common.h"

void	
Parameters::updateParameters(const OP_Inputs* in)
{
	colorBits = in->getParInt(COLORBITS_NAME);
    dither = in->getParInt(DITHER_NAME) ? true : false;
    multiThreaded = in->getParInt(MULTITHREADED_NAME) ? true : false;
    downloadType = static_cast<OP_TOPInputDownloadType>(in->getParInt(DOWNLOADTYPE_NAME));
}

void
Parameters::setupParameters(OP_ParameterManager* manager)
{
    {
		OP_NumericParameter np;
		np.name = COLORBITS_NAME;
		np.label = "Bits per Color";
		np.page = "Filter";

		np.defaultValues[0] = 2;
		np.maxSliders[0] = np.maxValues[0] = 8;
		np.minSliders[0] = np.minValues[0] = 1;
		np.clampMaxes[0] = true;
		np.clampMins[0] = true;

		OP_ParAppendResult res = manager->appendInt(np);
		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter np;
		np.name = DITHER_NAME;
		np.label = "Dither";
		np.page = "Filter";

		np.defaultValues[0] = true;

		OP_ParAppendResult res = manager->appendToggle(np);
		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter np;
		np.name = MULTITHREADED_NAME;
		np.label = "Multithreaded";
		np.page = "Filter";

		np.defaultValues[0] = false;

		OP_ParAppendResult res = manager->appendToggle(np);
		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter sp;
		sp.name = DOWNLOADTYPE_NAME;
		sp.label = "Download Type";
		sp.page = "Filter";

		const char* names[] = { "Delayed", "Instant" };
		const char* labels[] = { "Delayed", "Instant" };

		OP_ParAppendResult res = manager->appendMenu(sp, 2, names, labels);
		assert(res == OP_ParAppendResult::Success);
	}
}