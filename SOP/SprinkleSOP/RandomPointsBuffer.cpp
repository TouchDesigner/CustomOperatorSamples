#include "RandomPointsBuffer.h"

#include "CPlusPlus_Common.h"
#include <algorithm>

using namespace TD;

enum class Axis
{
	X = 0,
	Y = 1,
	Z = 2
};

namespace
{
	double dist(const Position& A, const Position& B)
	{
		Vector AB{ B.x - A.x, B.y - A.y, B.z - A.z };
		return AB.length();
	}
}

class Node
{
public:
	Node(size_t pos) : left{ nullptr }, right{ nullptr }, index{ pos }
	{
	}

	~Node()
	{
		delete left;
		delete right;
	}

	void 
	addPoint(std::vector<Position>& pts, const Position& pos, int axis)
	{
		double newVal;
		double curVal;

		switch (static_cast<Axis>(axis))
		{
			default:
			case Axis::X:
				newVal = pos.x;
				curVal = pts.at(index).x;
				break;
			case Axis::Y:
				newVal = pos.y;
				curVal = pts.at(index).y;
				break;
			case Axis::Z:
				newVal = pos.z;
				curVal = pts.at(index).z;
				break;
		}

		Node** next;
		if (newVal < curVal)
			next = &right;
		else
			next = &left;

		if (*next)
		{
			(*next)->addPoint(pts, pos, (axis + 1) % 3);
		}
		else
		{
			*next = new Node(pts.size());
			pts.push_back(pos);
		}
	}

	void 
	nearestNeighborDist(std::vector<Position>& pts, const Position& pos, const double& constraint , int axis, double& minDist)
	{
		double newVal;
		double curVal;

		switch (static_cast<Axis>(axis))
		{
		default:
		case Axis::X:
			newVal = pos.x;
			curVal = pts.at(index).x;
			break;
		case Axis::Y:
			newVal = pos.y;
			curVal = pts.at(index).y;
			break;
		case Axis::Z:
			newVal = pos.z;
			curVal = pts.at(index).z;
			break;
		}

		double planeDist = std::fabs(newVal - curVal);
		minDist = std::min(minDist, dist(pos, pts.at(index)));
		
		Node*	next;
		Node*	other;

		if (newVal < curVal)
		{
			next = right;
			other = left;
		}
		else
		{
			next = left;
			other = right;
		}

		if (minDist < constraint)
			return;

		if (planeDist < constraint && other)
		{
			other->nearestNeighborDist(pts, pos, constraint, (axis + 1) % 3, minDist);
		}

		if (next)
		{
			next->nearestNeighborDist(pts, pos, constraint, (axis + 1) % 3, minDist);
		}
	}

	Node*	left;
	Node*	right;
	size_t	index;
};

RandomPointsBuffer::RandomPointsBuffer(bool separatePoints, double distance) : 
	root{ nullptr }, mySeparatePoints{ separatePoints }, myDistance{ distance }
{
}


RandomPointsBuffer::~RandomPointsBuffer()
{
	delete root;
	points.clear();
}

bool 
RandomPointsBuffer::addIfPointFits(const Position& p)
{
	if (mySeparatePoints)
	{
		if (doesPointFit(p))
		{
			addPoint(p);
			return true;
		}
	}
	else
	{
		points.push_back(p);
		return true;
	}

	return false;
}

std::vector<Position>& 
RandomPointsBuffer::getVector()
{
	return points;
}

void RandomPointsBuffer::addPoint(const Position& p)
{
	if (!root)
	{
		root = new Node(0);
		points.push_back(p);
	}
	else
	{
		root->addPoint(points, p, static_cast<int>(Axis::X));
	}
}

bool 
RandomPointsBuffer::doesPointFit(const Position& p)
{
	if (!root)
		return true;

	double distance = dist(p, points.front());
	root->nearestNeighborDist(points, p, myDistance, static_cast<int>(Axis::X), distance);
	return distance > myDistance;
}
