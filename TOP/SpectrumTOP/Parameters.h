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

namespace TD
{
	class OP_Inputs;
	class OP_ParameterManager;
}

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char ModeName[] = "Mode";
constexpr static char ModeLabel[] = "Mode";

constexpr static char CoordName[] = "Coord";
constexpr static char CoordLabel[] = "Coordinate System";

constexpr static char ChanName[] = "Chan";
constexpr static char ChanLabel[] = "Channel";

constexpr static char TransrowsName[] = "Transrows";
constexpr static char TransrowsLabel[] = "Transform Rows";


#pragma endregion

#pragma region Menus
enum class ModeMenuItems
{
	dft,
	idft
};

enum class CoordMenuItems
{
	polar,
	cartesian
};

enum class ChanMenuItems
{
	r,
	g,
	b,
	a
};

#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Mode
	static ModeMenuItems		evalMode(const TD::OP_Inputs* inputs);

	// Coordinate System
	static CoordMenuItems		evalCoord(const TD::OP_Inputs* inputs);

	// Channel
	static ChanMenuItems		evalChan(const TD::OP_Inputs* inputs);

	// Transform Rows
	static bool		evalTransrows(const TD::OP_Inputs* inputs);


};
#pragma endregion