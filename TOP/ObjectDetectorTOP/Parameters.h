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

constexpr static char ClassifierName[] = "Classifier";
constexpr static char ClassifierLabel[] = "Classifier";

constexpr static char ScalefactorName[] = "Scalefactor";
constexpr static char ScalefactorLabel[] = "Scale Factor";

constexpr static char MinneighborsName[] = "Minneighbors";
constexpr static char MinneighborsLabel[] = "Min Neighbors";

constexpr static char LimitobjectsizeName[] = "Limitobjectsize";
constexpr static char LimitobjectsizeLabel[] = "Limit Object Size";

constexpr static char MinobjectwidthName[] = "Minobjectwidth";
constexpr static char MinobjectwidthLabel[] = "Min Object Width";

constexpr static char MinobjectheightName[] = "Minobjectheight";
constexpr static char MinobjectheightLabel[] = "Min Object Height";

constexpr static char MaxobjectwidthName[] = "Maxobjectwidth";
constexpr static char MaxobjectwidthLabel[] = "Max Object Width";

constexpr static char MaxobjectheightName[] = "Maxobjectheight";
constexpr static char MaxobjectheightLabel[] = "Max Object Height";

constexpr static char DrawboundingboxName[] = "Drawboundingbox";
constexpr static char DrawboundingboxLabel[] = "Draw Bounding Box";

constexpr static char LimitobjectsdetectedName[] = "Limitobjectsdetected";
constexpr static char LimitobjectsdetectedLabel[] = "Limit Objects Detected";

constexpr static char MaximumobjectsName[] = "Maximumobjects";
constexpr static char MaximumobjectsLabel[] = "Maximum Objects";


#pragma endregion

#pragma region Menus
#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Classifier
	static std::string		evalClassifier(const TD::OP_Inputs* inputs);

	// Scale Factor
	static double		evalScalefactor(const TD::OP_Inputs* inputs);

	// Min Neighbors
	static int		evalMinneighbors(const TD::OP_Inputs* inputs);

	// Limit Object Size
	static bool		evalLimitobjectsize(const TD::OP_Inputs* inputs);

	// Min Object Width
	static double		evalMinobjectwidth(const TD::OP_Inputs* inputs);

	// Min Object Height
	static double		evalMinobjectheight(const TD::OP_Inputs* inputs);

	// Max Object Width
	static double		evalMaxobjectwidth(const TD::OP_Inputs* inputs);

	// Max Object Height
	static double		evalMaxobjectheight(const TD::OP_Inputs* inputs);

	// Draw Bounding Box
	static bool		evalDrawboundingbox(const TD::OP_Inputs* inputs);

	// Limit Objects Detected
	static bool		evalLimitobjectsdetected(const TD::OP_Inputs* inputs);

	// Maximum Objects
	static int		evalMaximumobjects(const TD::OP_Inputs* inputs);


};
#pragma endregion