/* Shared Use License: This file is owned by Derivative Inc. (Derivative)
* and can only be used, and/or modified for use, in conjunction with
* Derivative's TouchDesigner software, and only if you are a licensee who has
* accepted Derivative's TouchDesigner license or assignment agreement
* (which also govern the use of this file). You may share or redistribute
* a modified version of this file provided the following conditions are met:
*
* 1. The shared file or redistribution must retain the information set out
* above and this list of conditions.
* 2. Derivative's name (Derivative Inc.) or its trademarks may not be used
* to endorse or promote products derived from this file without specific
* prior written permission from Derivative.
*/

#ifndef __VolSprinkleTree__
#define __VolSprinkleTree__

#include <vector>
#include <random>
#include "CPlusPlus_Common.h"

class SOP_Output;

struct BoxNode;

class VolSprinkleTree
{
public:
	VolSprinkleTree(const OP_SOPInput* sop, BoundingBox& bb);
	~VolSprinkleTree();

	Position getPoint(float, float, float, float);

	void outputTest(SOP_Output*);
private:
	constexpr static int	myBoxNumber = 50000;

	void processBox(BoundingBox&, const OP_SOPInput*);

	std::vector<BoxNode>	myTree;
	std::mt19937			myRNG;
};

#endif
