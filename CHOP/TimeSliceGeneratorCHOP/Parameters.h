#pragma once

#include<string>
#include "CPlusPlus_Common.h"

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
	static void		setup(TD::OP_ParameterManager*);

	// Type
	static TypeMenuItems		evalType(const TD::OP_Inputs* inputs);

	// Frequency
	static double		evalFrequency(const TD::OP_Inputs* inputs);

	// Apply Scale
	static bool		evalApplyscale(const TD::OP_Inputs* inputs);

	// Scale
	static double		evalScale(const TD::OP_Inputs* inputs);


};
#pragma endregion