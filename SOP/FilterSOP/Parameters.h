#pragma once

#include<string>

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char TranslatechopName[] = "Translatechop";
constexpr static char TranslatechopLabel[] = "Translate CHOP";


#pragma endregion

#pragma region Menus
#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Translate CHOP
	static const TD::OP_CHOPInput*		evalTranslatechop(const TD::OP_Inputs* inputs);


};
#pragma endregion