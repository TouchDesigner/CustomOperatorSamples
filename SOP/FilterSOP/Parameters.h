#pragma once

class OP_Inputs;
class OP_ParameterManager;

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
	static void		setup(OP_ParameterManager*);

	// Translate CHOP
	static const OP_CHOPInput*		evalTranslatechop(const OP_Inputs* input);


};
#pragma endregion