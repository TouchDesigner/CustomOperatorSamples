#pragma once

class OP_Inputs;
class OP_ParameterManager;

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char Color1Name[] = "Color1";
constexpr static char Color1Label[] = "Color 1";

constexpr static char Color2Name[] = "Color2";
constexpr static char Color2Label[] = "Color 2";

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

	// Color 1
	static Color		evalColor1(const OP_Inputs* input);

	// Color 2
	static Color		evalColor2(const OP_Inputs* input);

	// Speed
	static double		evalSpeed(const OP_Inputs* input);

	// Reset
	static int		evalReset(const OP_Inputs* input);


};
#pragma endregion