#pragma once

class OP_Inputs;
class OP_ParameterManager;

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
	static void		setup(OP_ParameterManager*);

	// Low Threshold
	static double		evalLowthreshold(const OP_Inputs* input);

	// High Threshold
	static double		evalHighthreshold(const OP_Inputs* input);

	// Apperture Size
	static int		evalApperturesize(const OP_Inputs* input);

	// L2 Gradient
	static bool		evalL2gradient(const OP_Inputs* input);


};
#pragma endregion