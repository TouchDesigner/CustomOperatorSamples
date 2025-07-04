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