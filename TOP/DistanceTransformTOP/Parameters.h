#pragma once

#include<string>
#include "CPlusPlus_Common.h"

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char DistancetypeName[] = "Distancetype";
constexpr static char DistancetypeLabel[] = "Distance Type";

constexpr static char MasksizeName[] = "Masksize";
constexpr static char MasksizeLabel[] = "Mask Size";

constexpr static char NormalizeName[] = "Normalize";
constexpr static char NormalizeLabel[] = "Normalize";

constexpr static char ChannelName[] = "Channel";
constexpr static char ChannelLabel[] = "Channel";


#pragma endregion

#pragma region Menus
enum class DistancetypeMenuItems
{
	L1,
	L2,
	C
};

enum class MasksizeMenuItems
{
	Three,
	Five,
	Precise
};

enum class ChannelMenuItems
{
	R,
	G,
	B,
	A
};

#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Distance Type
	static DistancetypeMenuItems		evalDistancetype(const TD::OP_Inputs* inputs);

	// Mask Size
	static MasksizeMenuItems		evalMasksize(const TD::OP_Inputs* inputs);

	// Normalize
	static bool		evalNormalize(const TD::OP_Inputs* inputs);

	// Channel
	static ChannelMenuItems		evalChannel(const TD::OP_Inputs* inputs);


};
#pragma endregion