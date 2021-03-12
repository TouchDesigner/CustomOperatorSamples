#pragma once

class OP_Inputs;
class OP_ParameterManager;

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char LengthName[] = "Length";
constexpr static char LengthLabel[] = "Length";

constexpr static char NumberofchannelsName[] = "Numberofchannels";
constexpr static char NumberofchannelsLabel[] = "Number of Channels";

constexpr static char ApplyscaleName[] = "Applyscale";
constexpr static char ApplyscaleLabel[] = "Apply Scale";

constexpr static char ScaleName[] = "Scale";
constexpr static char ScaleLabel[] = "Scale";

constexpr static char OperationName[] = "Operation";
constexpr static char OperationLabel[] = "Operation";


#pragma endregion

#pragma region Menus
enum class OperationMenuItems
{
	Add,
	Multiply,
	Power
};

#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(OP_ParameterManager*);

	// Length
	static int		evalLength(const OP_Inputs* input);

	// Number of Channels
	static int		evalNumberofchannels(const OP_Inputs* input);

	// Apply Scale
	static bool		evalApplyscale(const OP_Inputs* input);

	// Scale
	static double		evalScale(const OP_Inputs* input);

	// Operation
	static OperationMenuItems		evalOperation(const OP_Inputs* input);


};
#pragma endregion