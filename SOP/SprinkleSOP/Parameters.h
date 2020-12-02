#ifndef __Parameters__
#define __Parameters__

class OP_Inputs;
class OP_ParameterManager;

// Names of the parameters
constexpr static char	SEED_NAME[] = "Seed";
constexpr static char	GENERATE_NAME[] = "Generate";
constexpr static char	POINTCOUNT_NAME[] = "Pointcount";
constexpr static char	FORCEDISTANCE_NAME[] = "Forcedistance";
constexpr static char	POINTDISTANCE_NAME[] = "Pointdistance";

enum class Generate
{
	Density,
	Primitive,
	BoundingBox,
	Volume
};

struct Parameters
{
public:
	void	evalParms(const OP_Inputs*);

	void    setupParms(OP_ParameterManager*);

	double		seed;
	Generate	generate;
	int			pointcount;
	bool		forcedistance;
	double		pointdistance;

	bool		changed;
};

#endif
