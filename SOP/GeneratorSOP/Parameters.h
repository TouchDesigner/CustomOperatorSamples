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

constexpr static char ShapeName[] = "Shape";
constexpr static char ShapeLabel[] = "Shape";

constexpr static char ColorName[] = "Color";
constexpr static char ColorLabel[] = "Color";

constexpr static char GpudirectName[] = "Gpudirect";
constexpr static char GpudirectLabel[] = "GPU Direct";


#pragma endregion

#pragma region Menus
enum class ShapeMenuItems
{
	Point,
	Line,
	Square,
	Cube
};

#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Shape
	static ShapeMenuItems		evalShape(const TD::OP_Inputs* inputs);

	// Color
	static TD::Color		evalColor(const TD::OP_Inputs* inputs);

	// GPU Direct
	static bool		evalGpudirect(const TD::OP_Inputs* inputs);


};
#pragma endregion