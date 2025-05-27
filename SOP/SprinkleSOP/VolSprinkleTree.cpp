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

#include "VolSprinkleTree.h"
#include "CPlusPlus_Common.h"
#include "SOP_CPlusPlusBase.h"
#include <random>
#include <array>

using namespace TD;

namespace
{
	float getVolume(BoundingBox& bb)
	{
		return bb.sizeX() * bb.sizeY() * bb.sizeZ();
	}

	Position pointInBox(BoundingBox& bb, float r1, float r2, float r3)
	{
		Position p;
		p.x = r1 * bb.sizeX() + bb.minX;
		p.y = r2 * bb.sizeY() + bb.minY;
		p.z = r3 * bb.sizeZ() + bb.minZ;
		return p;
	}
}

struct BoxNode
{
	BoxNode(BoundingBox& bb) : box{ bb }, cumulativeVolume{}
	{
	}
	BoxNode(Position& min, Position& max) : box{ min, max }, cumulativeVolume{}
	{
	}

	operator float() const { return cumulativeVolume; }

	BoundingBox	box;
	float		cumulativeVolume;
};

VolSprinkleTree::VolSprinkleTree(const TD::OP_SOPInput* sop, BoundingBox& bb)
{
	float boxSize = std::cbrt(getVolume(bb) / myBoxNumber);
	Vector boxSizeV = Vector{ boxSize, boxSize, boxSize };

	for (float x = bb.minX; x < bb.maxX; x += boxSize)
	{
		for (float y = bb.minY; y < bb.maxY; y += boxSize)
		{
			for (float z = bb.minZ; z < bb.maxZ; z += boxSize)
			{
				Position origin = Position(x, y, z);
				BoundingBox boundingBox = BoundingBox(origin, origin + boxSizeV);
				processBox(boundingBox, sop);
			}
		}
	}

	if (myTree.empty())
		myTree.emplace_back(BoxNode(bb));

	float volume = 0.0;
	for (BoxNode& bn : myTree)
	{
		volume += getVolume(bn.box);
		bn.cumulativeVolume = volume;
	}
}

VolSprinkleTree::~VolSprinkleTree()
{
}

Position 
VolSprinkleTree::getPoint(float r1, float r2, float r3, float r4)
{
	float target = r1 * myTree.back().cumulativeVolume;
	auto it = std::upper_bound(myTree.begin(), myTree.end(), target);
	return pointInBox(it->box, r2, r3, r4);
}

void
VolSprinkleTree::processBox(BoundingBox& bb, const TD::OP_SOPInput* sop)
{
	Position	p;

	bb.getCenter(&p);
	if (((TD::OP_SOPInput*)sop)->isInside(p))
	{
		myTree.emplace_back(BoxNode(bb));
		return;
	}
}

void VolSprinkleTree::outputTest(TD::SOP_Output* out)
{
	auto outputBox = [&](BoundingBox& bb) {
		int idx = out->getNumPoints();
		Position points[] = {
			Position(bb.minX, bb.minY, bb.minZ),
			Position(bb.minX, bb.maxY, bb.minZ),
			Position(bb.minX, bb.minY, bb.maxZ),
			Position(bb.minX, bb.maxY, bb.maxZ),
			Position(bb.maxX, bb.minY, bb.maxZ),
			Position(bb.maxX, bb.maxY, bb.maxZ),
			Position(bb.maxX, bb.minY, bb.minZ),
			Position(bb.maxX, bb.maxY, bb.minZ) };
		out->addPoints(points, 8);
		int32_t primpoints[] = {
			idx, idx + 1, idx + 2,
			idx + 1, idx + 2, idx + 3,
			idx + 2, idx + 3, idx + 4,
			idx + 3, idx + 4, idx + 5,
			idx + 4, idx + 5, idx + 6,
			idx + 5, idx + 6, idx + 7,
			idx + 6, idx + 7, idx,
			idx + 7, idx, idx + 1,
			idx, idx + 2, idx + 6,
			idx + 2, idx + 4, idx + 6,
			idx + 1, idx + 3, idx + 5,
			idx + 1, idx + 5, idx + 7
		};
		out->addTriangles(primpoints, 12);
	};

	for (BoxNode& bn : myTree)
	{
		outputBox(bn.box);
	}
}
