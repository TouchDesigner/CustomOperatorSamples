#include "Parameters.h"
#include "CPlusPlus_Common.h"
#include <opencv2\core.hpp>

void
Parameters::evalParms(const OP_Inputs* input)
{
	transform = static_cast<Transform>(input->getParInt(TRANSFORM_NAME));
	channel = static_cast<Channel>(input->getParInt(CHANNEL_NAME));
	transformrows = input->getParInt(TRANSFORMROWS_NAME) ? true : false;
	coordinatesystem = static_cast<CoordinatesSys>(input->getParInt(COORDINATESYSTEM_NAME));

	flags = cv::DFT_COMPLEX_INPUT;
	flags |= transformrows ? cv::DFT_ROWS : 0;
	flags |= transform == Transform::Inverse ? cv::DFT_INVERSE | cv::DFT_SCALE : 0;

	input->enablePar(CHANNEL_NAME, transform == Transform::Forward);
}

void
Parameters::setupParms(OP_ParameterManager* manager)
{
	{
		OP_StringParameter p;
		p.name = TRANSFORM_NAME;
		p.label = "Transform";
		p.page = "DFT";
		p.defaultValue = "name1";

		const char*	names[] = { "Imagetodft", "Dfttoimage" };
		const char*	labels[] = { "Image To DFT", "DFT To Image" };
		OP_ParAppendResult res = manager->appendMenu(p, 2, names, labels);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter p;
		p.name = COORDINATESYSTEM_NAME;
		p.label = "Coordinate System";
		p.page = "DFT";
		p.defaultValue = "Polar";

		const char*	names[] = { "Polar", "Cartesian" };
		const char*	labels[] = { "Polar [Magnitude, Phase]", "Cartesian [Real, Imaginary]" };
		OP_ParAppendResult res = manager->appendMenu(p, 2, names, labels);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_StringParameter p;
		p.name = CHANNEL_NAME;
		p.label = "Channel";
		p.page = "DFT";
		p.defaultValue = "R";

		const char*	names[] = { "R", "G", "B", "A" };
		const char*	labels[] = { "R", "G", "B", "A" };
		OP_ParAppendResult res = manager->appendMenu(p, 4, names, labels);

		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter p;
		p.name = TRANSFORMROWS_NAME;
		p.label = "Per Row";
		p.page = "DFT";

		p.defaultValues[0] = false;
		OP_ParAppendResult res = manager->appendToggle(p);

		assert(res == OP_ParAppendResult::Success);
	}
}
