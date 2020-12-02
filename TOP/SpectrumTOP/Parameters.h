#ifndef __Parameters__
#define __Parameters__

// Names of the parameters
constexpr static char	TRANSFORM_NAME[] = "Transform";
constexpr static char	COORDINATESYSTEM_NAME[] = "Coordinatesystem";
constexpr static char	CHANNEL_NAME[] = "Channel";
constexpr static char	TRANSFORMROWS_NAME[] = "Transformrows";

class OP_Inputs;
class OP_ParameterManager;

enum class Transform
{
	Forward,
	Inverse
};

enum class CoordinatesSys
{
	Polar,
	Cartesian
};

enum class Channel
{
	R,
	Mono = R,
	G,
	Second = G,
	B,
	A,
	Invalid
};

struct Parameters
{
public:
	void	evalParms(const OP_Inputs*);

    void    setupParms(OP_ParameterManager*);

	// Parameters saved
	int				flags;
	bool			transformrows;
	CoordinatesSys	coordinatesystem;
	Transform		transform;
	Channel			channel;
};

#endif
