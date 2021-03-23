#pragma once

class OP_Inputs;
class OP_ParameterManager;

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char DistancetypeName[] = "Distancetype";
constexpr static char DistancetypeLabel[] = "Distance Type";

constexpr static char MasksizeName[] = "Masksize";
constexpr static char MasksizeLabel[] = "Mask Size";

constexpr static char NormalizeName[] = "Normalize";
constexpr static char NormalizeLabel[] = "Normalize";

constexpr static char DownloadtypeName[] = "Downloadtype";
constexpr static char DownloadtypeLabel[] = "Download Type";

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

enum class DownloadtypeMenuItems
{
	Delayed,
	Instant
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
	static void		setup(OP_ParameterManager*);

	// Distance Type
	static DistancetypeMenuItems		evalDistancetype(const OP_Inputs* input);

	// Mask Size
	static MasksizeMenuItems		evalMasksize(const OP_Inputs* input);

	// Normalize
	static bool		evalNormalize(const OP_Inputs* input);

	// Download Type
	static DownloadtypeMenuItems		evalDownloadtype(const OP_Inputs* input);

	// Channel
	static ChannelMenuItems		evalChannel(const OP_Inputs* input);


};
#pragma endregion