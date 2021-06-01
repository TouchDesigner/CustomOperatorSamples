#pragma once

class OP_Inputs;
class OP_ParameterManager;

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char MincutoffName[] = "Mincutoff";
constexpr static char MincutoffLabel[] = "Cutoff Frequency (Hz)";

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
	static void		setup(OP_ParameterManager*);

	// Cutoff Frequency (Hz)
	static double		evalMincutoff(const OP_Inputs* input);

	// Speed Coefficient
	static double		evalBeta(const OP_Inputs* input);

	// Slope Cutoff Frequency (Hz)
	static double		evalDcutoff(const OP_Inputs* input);


};
#pragma endregion