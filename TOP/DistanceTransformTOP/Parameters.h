#ifndef __Parameters__
#define __Parameters__

#include <cstdint>

class OP_Inputs;
class OP_ParameterManager;

// Names of the parameters
constexpr static char	DISTANCETYPE_NAME[] = "Distancetype";
constexpr static char	MASKSIZE_NAME[] = "Masksize";
constexpr static char	NORMALIZE_NAME[] = "Normalize";
constexpr static char	DOWNLOADTYPE_NAME[] = "Downloadtype";
constexpr static char   CHANNEL_NAME[] = "Channel";

enum class
	Distancetype
{
	L1,
	L2,
	C
};
enum class
	Masksize
{
	Three,
	Five,
	Precise,
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
enum class OP_TOPInputDownloadType : int32_t;

struct Parameters
{
public:
	// Returns true if parameters have changed since last call
	bool	        evalParms(const OP_Inputs*);

	void	        setupParms(OP_ParameterManager*);

	Distancetype	distancetype;
	Masksize	masksize;
	bool	        normalize;
	OP_TOPInputDownloadType     downloadtype;
        Channel         channel;
};

#endif
