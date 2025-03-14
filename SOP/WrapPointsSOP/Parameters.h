#pragma once

#include<string>

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