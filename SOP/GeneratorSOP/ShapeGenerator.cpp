#include "ShapeGenerator.h"
#include <array>

// Cube descriptors
const std::array<Position, ShapeGenerator::theCubeNumPts>
ShapeGenerator::theCubePos = {
	// front
	Position(-1.0f, -1.0f, 1.0f),
	Position(-1.0f, 1.0f, 1.0f),
	Position(1.0f, -1.0f, 1.0f),
	Position(1.0f, 1.0f, 1.0f),
	// back
	Position(-1.0f, -1.0f, -1.0f),
	Position(-1.0f, 1.0f, -1.0f),
	Position(1.0f, -1.0f, -1.0f),
	Position(1.0f, 1.0f, -1.0f),
	// top
	Position(-1.0f, 1.0f, -1.0f),
	Position(1.0f, 1.0f, -1.0f),
	Position(-1.0f, 1.0f, 1.0f),
	Position(1.0f, 1.0f, 1.0f),
	// bottom
	Position(-1.0f, -1.0f, -1.0f),
	Position(1.0f, -1.0f, -1.0f),
	Position(-1.0f, -1.0f, 1.0f),
	Position(1.0f, -1.0f, 1.0f),
	// right
	Position(1.0f, -1.0f, -1.0f),
	Position(1.0f, -1.0f, 1.0f),
	Position(1.0f, 1.0f, -1.0f),
	Position(1.0f, 1.0f, 1.0f),
	// left
	Position(-1.0f, -1.0f, -1.0f),
	Position(-1.0f, -1.0f, 1.0f),
	Position(-1.0f, 1.0f, -1.0f),
	Position(-1.0f, 1.0f, 1.0f)
};

const std::array<Vector, ShapeGenerator::theCubeNumPts>			
ShapeGenerator::theCubeNormals = {
	// front
	Vector(0.0f, 0.0f, 1.0f),
	Vector(0.0f, 0.0f, 1.0f),
	Vector(0.0f, 0.0f, 1.0f),
	Vector(0.0f, 0.0f, 1.0f),
	// back
	Vector(0.0f, 0.0f, -1.0f),
	Vector(0.0f, 0.0f, -1.0f),
	Vector(0.0f, 0.0f, -1.0f),
	Vector(0.0f, 0.0f, -1.0f),
	// top
	Vector(0.0f, 1.0f, 0.0f),
	Vector(0.0f, 1.0f, 0.0f),
	Vector(0.0f, 1.0f, 0.0f),
	Vector(0.0f, 1.0f, 0.0f),
	// bottom
	Vector(0.0f, -1.0f, 0.0f),
	Vector(0.0f, -1.0f, 0.0f),
	Vector(0.0f, -1.0f, 0.0f),
	Vector(0.0f, -1.0f, 0.0f),
	// right
	Vector(1.0f, 0.0f, 0.0f),
	Vector(1.0f, 0.0f, 0.0f),
	Vector(1.0f, 0.0f, 0.0f),
	Vector(1.0f, 0.0f, 0.0f),
	// left
	Vector(-1.0f, 0.0f, 0.0f),
	Vector(-1.0f, 0.0f, 0.0f),
	Vector(-1.0f, 0.0f, 0.0f),
	Vector(-1.0f, 0.0f, 0.0f)
};

const std::array<int32_t, ShapeGenerator::theCubeNumPrim * 3>	
ShapeGenerator::theCubeVertices = {
	// front
	0, 1, 2,
	3, 2, 1,
	// back
	6, 5, 4,
	5, 6, 7,
	// top
	8, 9, 10,
	11, 10, 9,
	// bottom
	14, 13, 12,
	13, 14, 15,
	// right
	16, 17, 18,
	19, 18, 17,
	// left
	22, 21, 20,
	21, 22, 23
};

const std::array<TexCoord, ShapeGenerator::theCubeNumPts>
ShapeGenerator::theCubeTexture = {
	// front
	TexCoord(2 / 3.0f, 0.0f, 0.0f),
	TexCoord(2 / 3.0f, 0.5f, 0.0f),
	TexCoord(3 / 3.0f, 0.0f, 0.0f),
	TexCoord(3 / 3.0f, 0.5f, 0.0f),
	// back
	TexCoord(0 / 3.0f, 0.5f, 0.0f),
	TexCoord(0 / 3.0f, 0.0f, 0.0f),
	TexCoord(1 / 3.0f, 0.5f, 0.0f),
	TexCoord(1 / 3.0f, 0.0f, 0.0f),
	// top
	TexCoord(2 / 3.0f, 1.0f, 0.0f),
	TexCoord(3 / 3.0f, 1.0f, 0.0f),
	TexCoord(2 / 3.0f, 0.5f, 0.0f),
	TexCoord(3 / 3.0f, 0.5f, 0.0f),
	// bottom
	TexCoord(1 / 3.0f, 0.5f, 0.0f),
	TexCoord(2 / 3.0f, 0.5f, 0.0f),
	TexCoord(1 / 3.0f, 1.0f, 0.0f),
	TexCoord(2 / 3.0f, 1.0f, 0.0f),
	// right
	TexCoord(2 / 3.0f, 0.0f, 0.0f),
	TexCoord(1 / 3.0f, 0.0f, 0.0f),
	TexCoord(2 / 3.0f, 0.5f, 0.0f),
	TexCoord(1 / 3.0f, 0.5f, 0.0f),
	// left
	TexCoord(1 / 3.0f, 1.0f, 0.0f),
	TexCoord(0 / 3.0f, 1.0f, 0.0f),
	TexCoord(1 / 3.0f, 0.5f, 0.0f),
	TexCoord(0 / 3.0f, 0.5f, 0.0f),
};

// Square descriptors					
const std::array<Position, ShapeGenerator::theSquareNumPts>		
ShapeGenerator::theSquarePos = {
	Position(-1.0f, -1.0f, 0.0f),
	Position(-1.0f, 1.0f, 0.0f),
	Position(1.0f, -1.0f, 0.0f),
	Position(1.0f, 1.0f, 0.0f)
};

const std::array<Vector, ShapeGenerator::theSquareNumPts>		
ShapeGenerator::theSquareNormals = {
	Vector(0.0f, 0.0f, 1.0f),
	Vector(0.0f, 0.0f, 1.0f),
	Vector(0.0f, 0.0f, 1.0f),
	Vector(0.0f, 0.0f, 1.0f)
};

const std::array<int32_t, ShapeGenerator::theSquareNumPrim * 3>	
ShapeGenerator::theSquareVertices = {
	0, 1, 2,
	3, 2, 1
};

const std::array<TexCoord, ShapeGenerator::theSquareNumPts>
ShapeGenerator::theSquareTexture = {
	TexCoord(0.0f, 0.0f, 0.0f),
	TexCoord(0.0f, 1.0f, 0.0f),
	TexCoord(1.0f, 0.0f, 0.0f),
	TexCoord(1.0f, 1.0f, 0.0f)
};

// Line descriptors						
const std::array<Position, ShapeGenerator::theLineNumPts>		
ShapeGenerator::theLinePos = {
	Position(-1.0f, -1.0f, -1.0f),
	Position(1.0f, 1.0f, 1.0f)
};

const std::array<Vector, ShapeGenerator::theLineNumPts>			
ShapeGenerator::theLineNormals = {
	Vector(-1.0f, 0.0f, 1.0f),
	Vector(-1.0f, 0.0f, 1.0f)
};

const std::array<int32_t, ShapeGenerator::theLineNumPts>			
ShapeGenerator::theLineVertices = {
	0, 1
};

const std::array<TexCoord, ShapeGenerator::theLineNumPts>
ShapeGenerator::theLineTexture = {
	TexCoord(0.0f, 0.0f, 0.0f),
	TexCoord(1.0f, 1.0f, 0.0f)
};

// Point descriptors
const Position													
ShapeGenerator::thePointPos = Position();

const Vector
ShapeGenerator::thePointNormal = Vector(0.0f, 0.0f, 1.0f);

const TexCoord
ShapeGenerator::thePointTexture = TexCoord();

void 
ShapeGenerator::outputDot(SOP_Output* output) const
{
	output->addPoint(thePointPos);
	output->setNormal(thePointNormal, 0);
	output->addParticleSystem(1, 0);
	output->setTexCoord(&thePointTexture, 1, 0);
}

void 
ShapeGenerator::outputLine(SOP_Output* output) const
{
	output->addPoints(theLinePos.data(), theLineNumPts);
	output->setNormals(theLineNormals.data(), theLineNumPts, 0);
	output->addLine(theLineVertices.data(), theLineNumPts);
	setPointTexCoords(output, theLineTexture.data(), theLineNumPts);
}

void 
ShapeGenerator::outputSquare(SOP_Output* output) const
{
	output->addPoints(theSquarePos.data(), theSquareNumPts);
	output->setNormals(theSquareNormals.data(), theSquareNumPts, 0);
	output->addTriangles(theSquareVertices.data(), theSquareNumPrim);
	setPointTexCoords(output, theSquareTexture.data(), theSquareNumPts);
}

void 
ShapeGenerator::outputCube(SOP_Output* output) const
{
	output->addPoints(theCubePos.data(), theCubeNumPts);
	output->setNormals(theCubeNormals.data(), theCubeNumPts, 0);
	output->addTriangles(theCubeVertices.data(), theCubeNumPrim);
	setPointTexCoords(output, theCubeTexture.data(), theCubeNumPts);
}

void 
ShapeGenerator::outputDotVBO(SOP_VBOOutput* output)
{
	output->allocVBO(1, 1, VBOBufferMode::Static);
	output->getPos()[0] = thePointPos;
	output->getNormals()[0] = thePointNormal;
	output->getTexCoords()[0] = thePointTexture;
	output->addParticleSystem(1)[0] = 0;
}

void 
ShapeGenerator::outputLineVBO(SOP_VBOOutput* output)
{
	output->allocVBO(theLineNumPts, theLineNumPts, VBOBufferMode::Static);
	myLastVBOAllocVertices = theLineNumPts;
	memcpy(output->getPos(), theLinePos.data(), theLineNumPts * sizeof(Position));
	memcpy(output->getNormals(), theLineNormals.data(), theLineNumPts * sizeof(Vector));
	memcpy(output->getTexCoords(), theLineTexture.data(), theLineNumPts * sizeof(TexCoord));
	memcpy(output->addLines(theLineNumPts), theLineVertices.data(), theLineNumPts * sizeof(int32_t));
}

void 
ShapeGenerator::outputSquareVBO(SOP_VBOOutput* output)
{
	output->allocVBO(theSquareNumPts, theSquareNumPrim * 3, VBOBufferMode::Static);
	myLastVBOAllocVertices = theSquareNumPts;
	memcpy(output->getPos(), theSquarePos.data(), theSquareNumPts * sizeof(Position));
	// Cannote memcpy normals since GPU is in the other direction
	Vector* outN = output->getNormals();
	for (int i = 0; i < theSquareNormals.size(); i += 1)
	{
		outN[i] = theSquareNormals.at(i) * -1.0f;
	}
	memcpy(output->getTexCoords(), theSquareTexture.data(), theSquareNumPts * sizeof(TexCoord));
	memcpy(output->addTriangles(theSquareNumPrim), theSquareVertices.data(), theSquareNumPts * 3 * sizeof(int32_t));
}

void 
ShapeGenerator::outputCubeVBO(SOP_VBOOutput* output)
{
	output->allocVBO(theCubeNumPts, theCubeNumPrim * 3, VBOBufferMode::Static);
	myLastVBOAllocVertices = theCubeNumPts;
	memcpy(output->getPos(), theCubePos.data(), theCubeNumPts * sizeof(Position));
	// Cannote memcpy normals since GPU is in the other direction
	Vector* outN = output->getNormals();
	for (int i = 0; i < theCubeNormals.size(); i += 1)
	{
		outN[i] = theCubeNormals.at(i) * -1.0f;
	}
	memcpy(output->getTexCoords(), theCubeTexture.data(), theCubeNumPts * sizeof(TexCoord));
	memcpy(output->addTriangles(theCubeNumPrim), theCubeVertices.data(), theCubeNumPrim * 3 * sizeof(int32_t));
}

int 
ShapeGenerator::getLastVBONumVertices() const
{
	return myLastVBOAllocVertices;
}

void 
ShapeGenerator::setPointTexCoords(SOP_Output* output, const TexCoord* t, int32_t numPts) const
{
	for (int i = 0; i < numPts; ++i)
	{
		output->setTexCoord(t + i, 1, i);
	}
}
