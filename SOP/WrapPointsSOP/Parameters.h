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

constexpr static char RaysName[] = "Rays";
constexpr static char RaysLabel[] = "Rays";

constexpr static char DirectionName[] = "Direction";
constexpr static char DirectionLabel[] = "Direction";

constexpr static char DestinationName[] = "Destination";
constexpr static char DestinationLabel[] = "Destination";

constexpr static char ReverseName[] = "Reverse";
constexpr static char ReverseLabel[] = "Reverse";

constexpr static char HitcolorName[] = "Hitcolor";
constexpr static char HitcolorLabel[] = "Hit Color";

constexpr static char MisscolorName[] = "Misscolor";
constexpr static char MisscolorLabel[] = "Miss Color";

constexpr static char ScaleName[] = "Scale";
constexpr static char ScaleLabel[] = "Scale";


#pragma endregion

#pragma region Menus
enum class RaysMenuItems
{
	Parallel,
	Radial
};

#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Rays
	static RaysMenuItems		evalRays(const TD::OP_Inputs* inputs);

	// Direction
	static std::array<double, 3>		evalDirection(const TD::OP_Inputs* inputs);

	// Destination
	static std::array<double, 3>		evalDestination(const TD::OP_Inputs* inputs);

	// Reverse
	static bool		evalReverse(const TD::OP_Inputs* inputs);

	// Hit Color
	static TD::Color		evalHitcolor(const TD::OP_Inputs* inputs);

	// Miss Color
	static TD::Color		evalMisscolor(const TD::OP_Inputs* inputs);

	// Scale
	static double		evalScale(const TD::OP_Inputs* inputs);


};
#pragma endregion