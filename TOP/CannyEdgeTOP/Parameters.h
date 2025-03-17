#pragma once

#include<string>

class TD::OP_Inputs;
class TD::OP_ParameterManager;

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char LowthresholdName[] = "Lowthreshold";
constexpr static char LowthresholdLabel[] = "Low Threshold";

constexpr static char HighthresholdName[] = "Highthreshold";
constexpr static char HighthresholdLabel[] = "High Threshold";

constexpr static char ApperturesizeName[] = "Apperturesize";
constexpr static char ApperturesizeLabel[] = "Apperture Size";

constexpr static char L2gradientName[] = "L2gradient";
constexpr static char L2gradientLabel[] = "L2 Gradient";


#pragma endregion

#pragma region Menus
#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Low Threshold
	static double		evalLowthreshold(const TD::OP_Inputs* inputs);

	// High Threshold
	static double		evalHighthreshold(const TD::OP_Inputs* inputs);

	// Apperture Size
	static int		evalApperturesize(const TD::OP_Inputs* inputs);

	// L2 Gradient
	static bool		evalL2gradient(const TD::OP_Inputs* inputs);


};
#pragma endregion