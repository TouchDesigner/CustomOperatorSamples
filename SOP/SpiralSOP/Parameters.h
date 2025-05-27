#pragma once

#include<string>

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char OrientationName[] = "Orientation";
constexpr static char OrientationLabel[] = "Orientation";

constexpr static char TopradiusName[] = "Topradius";
constexpr static char TopradiusLabel[] = "Top Radius";

constexpr static char BottomradiusName[] = "Bottomradius";
constexpr static char BottomradiusLabel[] = "Bottom Radius";

constexpr static char HeightName[] = "Height";
constexpr static char HeightLabel[] = "Height";

constexpr static char TurnsName[] = "Turns";
constexpr static char TurnsLabel[] = "Turns";

constexpr static char DivisionsName[] = "Divisions";
constexpr static char DivisionsLabel[] = "Divisions";

constexpr static char OutputgeometryName[] = "Outputgeometry";
constexpr static char OutputgeometryLabel[] = "Output Geometry";

constexpr static char StripwidthName[] = "Stripwidth";
constexpr static char StripwidthLabel[] = "Strip Width";

constexpr static char GpudirectName[] = "Gpudirect";
constexpr static char GpudirectLabel[] = "GPU Direct";


#pragma endregion

#pragma region Menus
enum class OrientationMenuItems
{
	X,
	Y,
	Z
};

enum class OutputgeometryMenuItems
{
	Points,
	Line,
	Trianglestrip
};

#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Orientation
	static OrientationMenuItems		evalOrientation(const TD::OP_Inputs* inputs);

	// Top Radius
	static double		evalTopradius(const TD::OP_Inputs* inputs);

	// Bottom Radius
	static double		evalBottomradius(const TD::OP_Inputs* inputs);

	// Height
	static double		evalHeight(const TD::OP_Inputs* inputs);

	// Turns
	static double		evalTurns(const TD::OP_Inputs* inputs);

	// Divisions
	static int		evalDivisions(const TD::OP_Inputs* inputs);

	// Output Geometry
	static OutputgeometryMenuItems		evalOutputgeometry(const TD::OP_Inputs* inputs);

	// Strip Width
	static double		evalStripwidth(const TD::OP_Inputs* inputs);

	// GPU Direct
	static bool		evalGpudirect(const TD::OP_Inputs* inputs);


};
#pragma endregion