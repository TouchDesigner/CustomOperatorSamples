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

constexpr static char RowsName[] = "Rows";
constexpr static char RowsLabel[] = "Rows";

constexpr static char ColumnsName[] = "Columns";
constexpr static char ColumnsLabel[] = "Columns";

constexpr static char LengthName[] = "Length";
constexpr static char LengthLabel[] = "Length";


#pragma endregion

#pragma region Menus
#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Seed
	static double		evalSeed(const TD::OP_Inputs* inputs);

	// Rows
	static int		evalRows(const TD::OP_Inputs* inputs);

	// Columns
	static int		evalColumns(const TD::OP_Inputs* inputs);

	// Length
	static int		evalLength(const TD::OP_Inputs* inputs);


};
#pragma endregion