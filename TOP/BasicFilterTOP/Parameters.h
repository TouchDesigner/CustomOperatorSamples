#pragma once

#include<string>

#pragma region ParNames and ParLabels

namespace TD {
	class OP_Inputs;
	class OP_ParameterManager;
}

// Names of the parameters

constexpr static char BitspercolorName[] = "Bitspercolor";
constexpr static char BitspercolorLabel[] = "Bits per Color";

constexpr static char DitherName[] = "Dither";
constexpr static char DitherLabel[] = "Dither";

constexpr static char MultithreadedName[] = "Multithreaded";
constexpr static char MultithreadedLabel[] = "Multithreaded";


#pragma endregion

#pragma region Menus
#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Bits per Color
	static int		evalBitspercolor(const TD::OP_Inputs* inputs);

	// Dither
	static bool		evalDither(const TD::OP_Inputs* inputs);

	// Multithreaded
	static bool		evalMultithreaded(const TD::OP_Inputs* inputs);


};
#pragma endregion