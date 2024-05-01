#pragma once

#include <array>
#include "CPlusPlus_Common.h"

class TD::OP_Inputs;
class TD::OP_ParameterManager;

#pragma region ParNames and ParLabels

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
	static RaysMenuItems		evalRays(const TD::OP_Inputs* input);

	// Direction
	static std::array<double, 3>		evalDirection(const TD::OP_Inputs* input);

	// Destination
	static std::array<double, 3>		evalDestination(const TD::OP_Inputs* input);

	// Reverse
	static bool		evalReverse(const TD::OP_Inputs* input);

	// Hit Color
	static TD::Color		evalHitcolor(const TD::OP_Inputs* input);

	// Miss Color
	static TD::Color		evalMisscolor(const TD::OP_Inputs* input);

	// Scale
	static double		evalScale(const TD::OP_Inputs* input);


};
#pragma endregion