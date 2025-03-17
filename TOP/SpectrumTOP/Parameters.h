#pragma once

#include<string>

class TD::OP_Inputs;
class TD::OP_ParameterManager;

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
	static void		setup(TD::OP_ParameterManager*);

	// Mode
	static ModeMenuItems		evalMode(const TD::OP_Inputs* inputs);

	// Coordinate System
	static CoordMenuItems		evalCoord(const TD::OP_Inputs* inputs);

	// Channel
	static ChanMenuItems		evalChan(const TD::OP_Inputs* inputs);

	// Transform Rows
	static bool		evalTransrows(const TD::OP_Inputs* inputs);


};
#pragma endregion