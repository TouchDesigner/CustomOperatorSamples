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

#ifndef __ShapeGenerator__
#define __ShapeGenerator__

#include <array>
#include "CPlusPlus_Common.h"
#include "SOP_CPlusPlusBase.h"

class ShapeGenerator
{
public:
	void	outputDot(SOP_Output*) const;

	void	outputLine(SOP_Output*) const;

	void	outputSquare(SOP_Output*) const;

	void	outputCube(SOP_Output*) const;

	// Output the shape directly to the GPU
	void	outputDotVBO(SOP_VBOOutput*);

	void	outputLineVBO(SOP_VBOOutput*);

	void	outputSquareVBO(SOP_VBOOutput*);

	void	outputCubeVBO(SOP_VBOOutput*);

	// Used to know how many verties we allocated last, 
	int		getLastVBONumVertices() const;

private:
	void	setPointTexCoords(SOP_Output*, const TexCoord* t, int numPts) const;

	// Cube descriptors 32 points 3 per  vertex
	constexpr static int									theCubeNumPts = 24;
	constexpr static int									theCubeNumPrim = 12;
	const static std::array<Position, theCubeNumPts>		theCubePos;
	const static std::array<Vector, theCubeNumPts>			theCubeNormals;
	const static std::array<int32_t, theCubeNumPrim * 3>	theCubeVertices;
	const static std::array<TexCoord, theCubeNumPts>		theCubeTexture;
											
	// Square descriptors
	constexpr static int									theSquareNumPts = 4;
	constexpr static int									theSquareNumPrim = 2;
	const static std::array<Position, theSquareNumPts>		theSquarePos;
	const static std::array<Vector, theSquareNumPts>		theSquareNormals;
	const static std::array<int32_t, theSquareNumPrim * 3>	theSquareVertices;
	const static std::array<TexCoord, theSquareNumPts>		theSquareTexture;
											
	// Line descriptors
	constexpr static int									theLineNumPts = 2;
	const static std::array<Position, theLineNumPts>		theLinePos;
	const static std::array<Vector, theLineNumPts>			theLineNormals;
	const static std::array<int32_t, theLineNumPts>			theLineVertices;
	const static std::array<TexCoord, theLineNumPts>		theLineTexture;

	// Point descriptors
	const static Position									thePointPos;
	const static Vector										thePointNormal;
	const static TexCoord									thePointTexture;

	// LastVBO allocation
	int myLastVBOAllocVertices;
};

#endif
