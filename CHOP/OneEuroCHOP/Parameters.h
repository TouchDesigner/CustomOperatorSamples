#pragma once

#include<string>

#pragma region ParNames and ParLabels

namespace TD {
	class OP_Inputs;
	class OP_ParameterManager;
}

// Names of the parameters

constexpr static char MincutoffName[] = "Mincutoff";
constexpr static char MincutoffLabel[] = "Cutoff Frequency(Hz)";

constexpr static char BetaName[] = "Beta";
constexpr static char BetaLabel[] = "Speed Coefficient";

constexpr static char DcutoffName[] = "Dcutoff";
constexpr static char DcutoffLabel[] = "Slope Cutoff Frequency (Hz)";


#pragma endregion

#pragma region Menus
#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Cutoff Frequency(Hz)
	static double		evalMincutoff(const TD::OP_Inputs* inputs);

	// Speed Coefficient
	static double		evalBeta(const TD::OP_Inputs* inputs);

	// Slope Cutoff Frequency (Hz)
	static double		evalDcutoff(const TD::OP_Inputs* inputs);


};
#pragma endregion