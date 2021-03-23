#ifndef __Parameters__
#define __Parameters__

#include <cstdint>

class OP_Inputs;
class OP_ParameterManager;

// Names of the parameters
constexpr static char	MODE_NAME[] = "Mode";
constexpr static char	METHOD_NAME[] = "Method";
constexpr static char	APPLYWATERSHED_NAME[] = "Applywatershed";
constexpr static char	SELECTOBJECT_NAME[] = "Selectobject";
constexpr static char	OBJECT_NAME[] = "Object";
constexpr static char	DOWNLOADTYPE_NAME[] = "Downloadtype";
constexpr static char   CHANNEL_NAME[] = "Channel";

enum class
	Mode
{
	External,
	List,
	Ccomp,
	Tree,
};

enum class
	Method
{
	None,
	Simple,
	Tcl1,
	Tckcos,
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
	bool	evalParms(const OP_Inputs*);

	void	setupParms(OP_ParameterManager*);

	Mode	mode;
	Method	method;
	bool	applywatershed;
	bool	selectobject;
	int	object;
	OP_TOPInputDownloadType     downloadtype;
        Channel channel;
};

#endif
