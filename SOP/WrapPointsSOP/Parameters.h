#ifndef __Parameters__
#define __Parameters__

#include "CPlusPlus_Common.h"

class OP_Inputs;
class OP_ParameterManager;

// Names of the parameters
constexpr static char	RAYS_NAME[] = "Rays";
constexpr static char	DIRECTION_NAME[] = "Direction";
constexpr static char	ORIGIN_NAME[] = "Destination";
constexpr static char	REVERSE_NAME[] = "Reverse";
constexpr static char	HITCOLOR_NAME[] = "Hitcolor";
constexpr static char	MISSCOLOR_NAME[] = "Misscolor";
constexpr static char	SCALE_NAME[] = "Scale";

enum class
Rays
{
	Parallel,
	Radial,
};

struct Parameters
{
public:
	// Returns true if parameters have changed since last call
	bool	evalParms(const OP_Inputs*);

	void	setupParms(OP_ParameterManager*);

	Rays	rays;
	double	direction[3];
	double	origin[3];
	bool	reverse;
	Color	hitcolor;
	Color	misscolor;
	double	scale;
};

#endif
