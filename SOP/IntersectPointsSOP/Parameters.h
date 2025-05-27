#pragma once

#include<string>

#pragma region ParNames and ParLabels

namespace TD {
	class OP_Inputs;
	class OP_ParameterManager;
}

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
	static void		setup(TD::OP_ParameterManager*);

	// Inside Color
	static TD::Color		evalInsidecolor(const TD::OP_Inputs* inputs);

	// Outside Color
	static TD::Color		evalOutsidecolor(const TD::OP_Inputs* inputs);


};
#pragma endregion