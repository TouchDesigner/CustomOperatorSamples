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

constexpr static char SeedName[] = "Seed";
constexpr static char SeedLabel[] = "Seed";

constexpr static char GenerateName[] = "Generate";
constexpr static char GenerateLabel[] = "Generate";

constexpr static char PointcountName[] = "Pointcount";
constexpr static char PointcountLabel[] = "Point Count";

constexpr static char SeparatepointsName[] = "Separatepoints";
constexpr static char SeparatepointsLabel[] = "Separate Points";

constexpr static char MinimumdistanceName[] = "Minimumdistance";
constexpr static char MinimumdistanceLabel[] = "Minimum Distance";


#pragma endregion

#pragma region Menus
enum class GenerateMenuItems
{
	Area,
	Primitive,
	Boundingbox,
	Volume
};

#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Seed
	static double		evalSeed(const TD::OP_Inputs* inputs);

	// Generate
	static GenerateMenuItems		evalGenerate(const TD::OP_Inputs* inputs);

	// Point Count
	static int		evalPointcount(const TD::OP_Inputs* inputs);

	// Separate Points
	static bool		evalSeparatepoints(const TD::OP_Inputs* inputs);

	// Minimum Distance
	static double		evalMinimumdistance(const TD::OP_Inputs* inputs);


};
#pragma endregion