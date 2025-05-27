#pragma once

#include<string>

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char ApplyscaleName[] = "Applyscale";
constexpr static char ApplyscaleLabel[] = "Apply Scale";

constexpr static char ScaleName[] = "Scale";
constexpr static char ScaleLabel[] = "Scale";

constexpr static char ApplyoffsetName[] = "Applyoffset";
constexpr static char ApplyoffsetLabel[] = "Apply Offset";

constexpr static char OffsetName[] = "Offset";
constexpr static char OffsetLabel[] = "Offset";


#pragma endregion

#pragma region Menus
#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Apply Scale
	static bool		evalApplyscale(const TD::OP_Inputs* inputs);

	// Scale
	static double		evalScale(const TD::OP_Inputs* inputs);

	// Apply Offset
	static bool		evalApplyoffset(const TD::OP_Inputs* inputs);

	// Offset
	static double		evalOffset(const TD::OP_Inputs* inputs);


};
#pragma endregion