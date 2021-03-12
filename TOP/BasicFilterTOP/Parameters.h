#pragma once

class OP_Inputs;
class OP_ParameterManager;

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char BitspercolorName[] = "Bitspercolor";
constexpr static char BitspercolorLabel[] = "Bits per Color";

constexpr static char DitherName[] = "Dither";
constexpr static char DitherLabel[] = "Dither";

constexpr static char MultithreadedName[] = "Multithreaded";
constexpr static char MultithreadedLabel[] = "Multithreaded";

constexpr static char DownloadtypeName[] = "Downloadtype";
constexpr static char DownloadtypeLabel[] = "Download Type";


#pragma endregion

#pragma region Menus
enum class DownloadtypeMenuItems
{
	Delayed,
	Instant
};

#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(OP_ParameterManager*);

	// Bits per Color
	static int		evalBitspercolor(const OP_Inputs* input);

	// Dither
	static bool		evalDither(const OP_Inputs* input);

	// Multithreaded
	static bool		evalMultithreaded(const OP_Inputs* input);

	// Download Type
	static DownloadtypeMenuItems		evalDownloadtype(const OP_Inputs* input);


};
#pragma endregion