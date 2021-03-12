#pragma once

class OP_Inputs;
class OP_ParameterManager;

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char BrightnessName[] = "Brightness";
constexpr static char BrightnessLabel[] = "Brightness";

constexpr static char SpeedName[] = "Speed";
constexpr static char SpeedLabel[] = "Speed";

constexpr static char ResetName[] = "Reset";
constexpr static char ResetLabel[] = "Reset";


#pragma endregion

#pragma region Menus
#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(OP_ParameterManager*);

	// Brightness
	static double		evalBrightness(const OP_Inputs* input);

	// Speed
	static double		evalSpeed(const OP_Inputs* input);

	// Reset
	static int		evalReset(const OP_Inputs* input);


};
#pragma endregion