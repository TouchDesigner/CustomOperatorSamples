#ifndef __FilterWork__
#define __FilterWork__
#include <cstdint>

class OP_Inputs;
struct Parameters;

namespace Filter
{
	void doFilterWork(uint32_t* inBuffer, int inWidth, int inHeight, uint32_t* outBuffer, int outWidth, int outHeight, Parameters parms);
}

#endif // !__FilterWork__

