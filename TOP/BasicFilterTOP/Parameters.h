#ifndef __Parameters__
#define __Parameters__

#include <cstdint>

// Names of the parameters
constexpr static char COLORBITS_NAME[] = "Colorbits";
constexpr static char DITHER_NAME[] = "Dither";
constexpr static char MULTITHREADED_NAME[] = "Multithreaded";
constexpr static char DOWNLOADTYPE_NAME[] = "Downloadtype";

class OP_Inputs;
class OP_ParameterManager;
enum class OP_TOPInputDownloadType : int32_t;

struct Parameters
{
public:
	void	updateParameters(const OP_Inputs*);

    void    setupParameters(OP_ParameterManager*);

	// Parameters saved
	int						    colorBits;
	bool						dither;
	bool						multiThreaded;
	OP_TOPInputDownloadType     downloadType;
};

#endif
