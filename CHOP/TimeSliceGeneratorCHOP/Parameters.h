/* Shared Use License: This file is owned by Derivative Inc. (Derivative)
* and can only be used, and/or modified for use, in conjunction with
* Derivative's TouchDesigner software, and only if you are a licensee who has
* accepted Derivative's TouchDesigner license or assignment agreement
* (which also govern the use of this file). You may share or redistribute
* a modified version of this file provided the following conditions are met:
*
* 1. The shared file or redistribution must retain the information set out
* above and this list of conditions.
* 2. Derivative's name (Derivative Inc.) or its trademarks may not be used
* to endorse or promote products derived from this file without specific
* prior written permission from Derivative.
*/

// Parameters.h generated using the cppParsTemplateGen Palette Component.
// https://derivative.ca/UserGuide/Palette:cppParsTemplateGen

#pragma once

#include<string>

#pragma region ParNames and ParLabels

namespace TD 
{
	class OP_Inputs;
	class OP_ParameterManager;
}

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