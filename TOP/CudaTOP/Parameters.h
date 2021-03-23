#pragma once

class OP_Inputs;
class OP_ParameterManager;

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char ColorName[] = "Color";
constexpr static char ColorLabel[] = "Color";


#pragma endregion

#pragma region Menus
#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(OP_ParameterManager*);

	// Color
	static Color		evalColor(const OP_Inputs* input);


};
#pragma endregion