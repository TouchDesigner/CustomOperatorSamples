#pragma once

#include<string>

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char LengthName[] = "Length";
constexpr static char LengthLabel[] = "Length";

constexpr static char NumberofchannelsName[] = "Numberofchannels";
constexpr static char NumberofchannelsLabel[] = "Number Of Channels";

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
	static void		setup(TD::OP_ParameterManager*);

	// Length
	static int		evalLength(const TD::OP_Inputs* inputs);

	// Number Of Channels
	static int		evalNumberofchannels(const TD::OP_Inputs* inputs);

	// Apply Scale
	static bool		evalApplyscale(const TD::OP_Inputs* inputs);

	// Scale
	static double		evalScale(const TD::OP_Inputs* inputs);

	// Operation
	static OperationMenuItems		evalOperation(const TD::OP_Inputs* inputs);


};
#pragma endregion