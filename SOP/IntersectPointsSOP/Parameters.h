#ifndef __Parameters__
#define __Parameters__

class OP_Inputs;
class OP_ParameterManager;

// Names of the parameters
constexpr static char	INSIDEVALUE_NAME[] = "Insidecolor";
constexpr static char	OUTSIDEVALUE_NAME[] = "Outsidecolor";


struct Parameters
{
public:
	// Returns true if parameters have changed since last call
	bool	evalParms(const OP_Inputs*);

	void	setupParms(OP_ParameterManager*);

	double	insidevalue[4];
	double	outsidevalue[4];
};

#endif
