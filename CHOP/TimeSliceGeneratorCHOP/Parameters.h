#pragma once

class OP_Inputs;
class OP_ParameterManager;

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char TypeName[] = "Type";
constexpr static char TypeLabel[] = "Type";

constexpr static char FrequencyName[] = "Frequency";
constexpr static char FrequencyLabel[] = "Frequency";

constexpr static char ApplyscaleName[] = "Applyscale";
constexpr static char ApplyscaleLabel[] = "Apply Scale";

constexpr static char ScaleName[] = "Scale";
constexpr static char ScaleLabel[] = "Scale";


#pragma endregion

#pragma region Menus
enum class TypeMenuItems
{
	Sine,
	Square,
	Ramp
};

#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(OP_ParameterManager*);

	// Type
	static TypeMenuItems		evalType(const OP_Inputs* input);

	// Frequency
	static double		evalFrequency(const OP_Inputs* input);

	// Apply Scale
	static bool		evalApplyscale(const OP_Inputs* input);

	// Scale
	static double		evalScale(const OP_Inputs* input);


};
#pragma endregion