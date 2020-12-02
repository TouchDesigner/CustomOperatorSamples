#ifndef __Parameters__
#define __Parameters__

class OP_Inputs;
class OP_ParameterManager;

// Names of the parameters
constexpr static char	LOWTHRESHOLD_NAME[] = "Lowthreshold";
constexpr static char	HIGHTHRESHOLD_NAME[] = "Highthreshold";
constexpr static char	APPERTURESIZE_NAME[] = "Apperturesize";
constexpr static char	L2GRADIENT_NAME[] = "L2gradient";


struct Parameters
{
public:
	// Returns true if parameters have changed since last call
	bool	evalParms(const OP_Inputs*);

	void	setupParms(OP_ParameterManager*);

	double	lowthreshold;
	double	highthreshold;
	int	apperturesize;
	bool	l2gradient;
};

#endif
