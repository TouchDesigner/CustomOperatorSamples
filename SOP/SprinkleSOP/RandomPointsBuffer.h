#ifndef __MinDistanceTree__
#define __MinDistanceTree__

#include <vector>

class Position;
class Node;

class RandomPointsBuffer
{
public:
	RandomPointsBuffer(bool separatePoints, double distance);

	~RandomPointsBuffer();

	bool	addIfPointFits(const Position&);

	std::vector<Position>&	getVector();

private:
	void	addPoint(const Position&);

	bool	doesPointFit(const Position&);

	Node*					root;
	std::vector<Position>	points;
	bool					mySeparatePoints;
	double					myDistance;
};

#endif
