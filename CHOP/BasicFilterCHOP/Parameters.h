#pragma once

class OP_Inputs;
class OP_ParameterManager;

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
	static void		setup(OP_ParameterManager*);

	// Apply Scale
	static bool		evalApplyscale(const OP_Inputs* input);

	// Scale
	static double		evalScale(const OP_Inputs* input);

	// Apply Offset
	static bool		evalApplyoffset(const OP_Inputs* input);

	// Offset
	static double		evalOffset(const OP_Inputs* input);


};
#pragma endregion