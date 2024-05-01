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
	void	outputDot(TD::SOP_Output*) const;

	void	outputLine(TD::SOP_Output*) const;

	void	outputSquare(TD::SOP_Output*) const;

	void	outputCube(TD::SOP_Output*) const;

	// Output the shape directly to the GPU
	void	outputDotVBO(TD::SOP_VBOOutput*);

	void	outputLineVBO(TD::SOP_VBOOutput*);

	void	outputSquareVBO(TD::SOP_VBOOutput*);

	void	outputCubeVBO(TD::SOP_VBOOutput*);

	// Used to know how many verties we allocated last, 
	int		getLastVBONumVertices() const;

private:
	void	setPointTexCoords(TD::SOP_Output*, const TD::TexCoord* t, int numPts) const;

	// Cube descriptors 32 points 3 per  vertex
	constexpr static int									theCubeNumPts = 24;
	constexpr static int									theCubeNumPrim = 12;
	const static std::array<TD::Position, theCubeNumPts>		theCubePos;
	const static std::array<TD::Vector, theCubeNumPts>			theCubeNormals;
	const static std::array<int32_t, theCubeNumPrim * 3>	theCubeVertices;
	const static std::array<TD::TexCoord, theCubeNumPts>		theCubeTexture;
											
	// Square descriptors
	constexpr static int									theSquareNumPts = 4;
	constexpr static int									theSquareNumPrim = 2;
	const static std::array<TD::Position, theSquareNumPts>		theSquarePos;
	const static std::array<TD::Vector, theSquareNumPts>		theSquareNormals;
	const static std::array<int32_t, theSquareNumPrim * 3>	theSquareVertices;
	const static std::array<TD::TexCoord, theSquareNumPts>		theSquareTexture;
											
	// Line descriptors
	constexpr static int									theLineNumPts = 2;
	const static std::array<TD::Position, theLineNumPts>		theLinePos;
	const static std::array<TD::Vector, theLineNumPts>			theLineNormals;
	const static std::array<int32_t, theLineNumPts>			theLineVertices;
	const static std::array<TD::TexCoord, theLineNumPts>		theLineTexture;

	// Point descriptors
	const static TD::Position									thePointPos;
	const static TD::Vector										thePointNormal;
	const static TD::TexCoord									thePointTexture;

	// LastVBO allocation
	int myLastVBOAllocVertices;
};

#endif
