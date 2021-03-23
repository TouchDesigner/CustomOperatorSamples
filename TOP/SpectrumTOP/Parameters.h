#pragma once

class OP_Inputs;
class OP_ParameterManager;

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char ModeName[] = "Mode";
constexpr static char ModeLabel[] = "Mode";

constexpr static char CoordName[] = "Coord";
constexpr static char CoordLabel[] = "Coordinate System";

constexpr static char ChanName[] = "Chan";
constexpr static char ChanLabel[] = "Channel";

constexpr static char TransrowsName[] = "Transrows";
constexpr static char TransrowsLabel[] = "Transform Rows";


#pragma endregion

#pragma region Menus
enum class ModeMenuItems
{
	dft,
	idft
};

enum class CoordMenuItems
{
	polar,
	cartesian
};

enum class ChanMenuItems
{
	r,
	g,
	b,
	a
};

#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(OP_ParameterManager*);

	// Mode
	static ModeMenuItems		evalMode(const OP_Inputs* input);

	// Coordinate System
	static CoordMenuItems		evalCoord(const OP_Inputs* input);

	// Channel
	static ChanMenuItems		evalChan(const OP_Inputs* input);

	// Transform Rows
	static bool		evalTransrows(const OP_Inputs* input);


};
#pragma endregion