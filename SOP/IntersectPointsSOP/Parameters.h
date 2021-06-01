#pragma once

class OP_Inputs;
class OP_ParameterManager;

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char InsidecolorName[] = "Insidecolor";
constexpr static char InsidecolorLabel[] = "Inside Color";

constexpr static char OutsidecolorName[] = "Outsidecolor";
constexpr static char OutsidecolorLabel[] = "Outside Color";


#pragma endregion

#pragma region Menus
#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(OP_ParameterManager*);

	// Inside Color
	static Color		evalInsidecolor(const OP_Inputs* input);

	// Outside Color
	static Color		evalOutsidecolor(const OP_Inputs* input);


};
#pragma endregion