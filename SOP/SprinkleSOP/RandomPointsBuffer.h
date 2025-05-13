#ifndef __MinDistanceTree__
#define __MinDistanceTree__

#include <vector>
#include "CPlusPlus_Common.h"

// class TD::Position;
class Node;

class RandomPointsBuffer
{
public:
	RandomPointsBuffer(bool separatePoints, double distance);

	~RandomPointsBuffer();

	bool	addIfPointFits(const TD::Position&);

	std::vector<TD::Position>&	getVector();

private:
	void	addPoint(const TD::Position&);

	bool	doesPointFit(const TD::Position&);

	Node*					root;
	std::vector<TD::Position>	points;
	bool					mySeparatePoints;
	double					myDistance;
};

#endif
