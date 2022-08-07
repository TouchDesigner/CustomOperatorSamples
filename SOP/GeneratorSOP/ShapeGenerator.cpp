#include "ShapeGenerator.h"
#include <array>
#include <vector>

#include "voronoi/voro++.hh"


using namespace voro;


// square distance
static float square_distance(Position a1, Position a2) {
	return (a1.x - a2.x) * (a1.x - a2.x) +
		(a1.y - a2.y) * (a1.y - a2.y) +
		(a1.z - a2.z) * (a1.z - a2.z);
	// while (--dims >= 0) {
}

static Position scaled(Position center, Position pos, float scale) {
	return Position(
		center.x * (1.0f - scale) + pos.x * scale,
		center.y * (1.0f - scale) + pos.y * scale,
		center.z * (1.0f - scale) + pos.z * scale
	);
}



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

void	ShapeGenerator::outputKDTree(const	OP_CHOPInput* input,
																	 float scale,
																	 SOP_Output* output) const
{
	// 0 - divide by x
	// 1 - divide by y
	// 2 - divide by z
	int dimension = 0;

	theCube initial = { 
		dimension,
		Position(0.f, 0.f, 0.f),
		{ Position(-1.0f,  1.0f, -1.0f),
			Position(-1.0f,  1.0f,  1.0f),
			Position(-1.0f, -1.0f,  1.0f),
			Position(-1.0f, -1.0f, -1.0f),

			Position( 1.0f,  1.0f, -1.0f),
			Position( 1.0f,  1.0f,  1.0f),
			Position( 1.0f, -1.0f,  1.0f),
			Position( 1.0f, -1.0f, -1.0f)
		} 
	};

	std::vector<theCube> theCubesFinal;
	std::vector<theCube> theCubes;
	theCubes.push_back(initial);
	
	// read points positions from input chop
	std::vector<Position> samples;
	for (int i = 0; i < input->numSamples; i++) {
		Position sample(
			input->getChannelData(0)[i],
			input->getChannelData(1)[i],
			input->getChannelData(2)[i]
		);
		samples.push_back(sample);
	}


	while (samples.size()) {
		
		int size = samples.size();
		
		// we will save processed cubes here for the next cycle...
		std::vector<theCube> theCubesNext;

		for (auto cubesIt = theCubes.begin(); cubesIt != theCubes.end(); cubesIt++) {

			// ************************************* //
			// find the point closest to cube center //
			// ************************************* //
			bool found(false);
			float minDistance(5.0f);
			std::vector<Position>::iterator middleSamplesIt = samples.begin();
			
			for (auto samplesIt = samples.begin(); samplesIt != samples.end(); samplesIt++)
			{
				if ( // this sample is inside our box
					(cubesIt->coords[3].x <= samplesIt->x) &&
					(cubesIt->coords[3].y <= samplesIt->y) &&
					(cubesIt->coords[3].z <= samplesIt->z) &&
					(cubesIt->coords[5].x >= samplesIt->x) &&
					(cubesIt->coords[5].y >= samplesIt->y) &&
					(cubesIt->coords[5].z >= samplesIt->z)) {

					found = true;

					// is our sample closer to center?
					float currentDistance = square_distance(*samplesIt, cubesIt->center);
					if (currentDistance < minDistance) {
						minDistance = currentDistance;
						middleSamplesIt = samplesIt;
					}
				} // if inside cube
			} // for samples
			

			// there are no samples/points in this cube - push it to the FINALcubes
			if (!found) {
				theCubesFinal.push_back(*cubesIt);
			
			// divide this cube, and create two cubes for the next cycle
			} else {

				// save the sample and erase it from the input array
				Position middleSample = *middleSamplesIt;
				samples.erase(middleSamplesIt);
				// the cube will be divided to two by sample point
				theCube left, right;

				switch (dimension)
				{
				// divide by X coordinate
				case 0: 
					left = {
						dimension,
						Position {
							(middleSample.x       + cubesIt->coords[3].x) / 2,
							(cubesIt->coords[5].y + cubesIt->coords[3].y) / 2,
							(cubesIt->coords[5].z + cubesIt->coords[3].z) / 2
						},
						{ cubesIt->coords[0],
							cubesIt->coords[1],
							cubesIt->coords[2],
							cubesIt->coords[3],
							Position(middleSample.x, cubesIt->coords[4].y, cubesIt->coords[4].z),
							Position(middleSample.x, cubesIt->coords[5].y, cubesIt->coords[5].z),
							Position(middleSample.x, cubesIt->coords[6].y, cubesIt->coords[6].z),
							Position(middleSample.x, cubesIt->coords[7].y, cubesIt->coords[7].z)
						}
					};
					right = {
						dimension,
						Position {
							(cubesIt->coords[5].x +       middleSample.x) / 2,
							(cubesIt->coords[5].y + cubesIt->coords[3].y) / 2,
							(cubesIt->coords[5].z + cubesIt->coords[3].z) / 2
						},
						{ Position(middleSample.x, cubesIt->coords[0].y, cubesIt->coords[0].z),
							Position(middleSample.x, cubesIt->coords[1].y, cubesIt->coords[1].z),
							Position(middleSample.x, cubesIt->coords[2].y, cubesIt->coords[2].z),
							Position(middleSample.x, cubesIt->coords[3].y, cubesIt->coords[3].z),
							cubesIt->coords[4],
							cubesIt->coords[5],
							cubesIt->coords[6],
							cubesIt->coords[7]
						}
					};
					break;
				// divide by Y coordinate
				case 1:
					left = { // top half
						dimension,
						Position {
							(cubesIt->coords[5].x + cubesIt->coords[3].x) / 2,
							(middleSample.y       + cubesIt->coords[3].y) / 2,
							(cubesIt->coords[5].z + cubesIt->coords[3].z) / 2
						},
						{ cubesIt->coords[0],
							cubesIt->coords[1],
							Position(cubesIt->coords[2].x, middleSample.y, cubesIt->coords[2].z),
							Position(cubesIt->coords[3].x, middleSample.y, cubesIt->coords[3].z),
							cubesIt->coords[4],
							cubesIt->coords[5],
							Position(cubesIt->coords[6].x, middleSample.y, cubesIt->coords[6].z),
							Position(cubesIt->coords[7].x, middleSample.y, cubesIt->coords[7].z)
						}
					};
					right = { // bottom half
						dimension,
						Position {
							(cubesIt->coords[5].x + cubesIt->coords[3].x) / 2,
							(cubesIt->coords[5].y +       middleSample.y) / 2,
							(cubesIt->coords[5].z + cubesIt->coords[3].z) / 2
						},
						{ 
							Position(cubesIt->coords[0].x, middleSample.y, cubesIt->coords[0].z),
							Position(cubesIt->coords[1].x, middleSample.y, cubesIt->coords[1].z),
							cubesIt->coords[2],
							cubesIt->coords[3],
							Position(cubesIt->coords[4].x, middleSample.y, cubesIt->coords[4].z),
							Position(cubesIt->coords[5].x, middleSample.y, cubesIt->coords[5].z),
							cubesIt->coords[6],
							cubesIt->coords[7]
						}
					};
					break;
				// divide by Z coordinate
				case 2:
				default:
					left = { // back half
						dimension,
						Position {
							(cubesIt->coords[5].x + cubesIt->coords[3].x) / 2,
							(cubesIt->coords[5].y + cubesIt->coords[3].y) / 2,
							(middleSample.z       + cubesIt->coords[3].z) / 2
						},
						{ cubesIt->coords[0],
							Position(cubesIt->coords[1].x, cubesIt->coords[1].y, middleSample.z),
							Position(cubesIt->coords[2].x, cubesIt->coords[2].y, middleSample.z),
							cubesIt->coords[3],
							cubesIt->coords[4],
							Position(cubesIt->coords[5].x, cubesIt->coords[5].y, middleSample.z),
							Position(cubesIt->coords[6].x, cubesIt->coords[6].y, middleSample.z),
							cubesIt->coords[7]
						}
					};
					right = { // front half
						dimension,
						Position {
							(cubesIt->coords[5].x + cubesIt->coords[3].x) / 2,
							(cubesIt->coords[5].y + cubesIt->coords[3].y) / 2,
							(cubesIt->coords[5].z +       middleSample.z) / 2
						},
						{	Position(cubesIt->coords[0].x, cubesIt->coords[0].y, middleSample.z),
							cubesIt->coords[1],
							cubesIt->coords[2],
							Position(cubesIt->coords[3].x, cubesIt->coords[3].y, middleSample.z),
							Position(cubesIt->coords[4].x, cubesIt->coords[4].y, middleSample.z),
							cubesIt->coords[5],
							cubesIt->coords[6],
							Position(cubesIt->coords[7].x, cubesIt->coords[7].y, middleSample.z)
						}
					};
					break;
				}

				// move the cubes to final if no samples left
				if (samples.size()) {
					theCubesNext.push_back(left);
					theCubesNext.push_back(right);
				} else {
					theCubesFinal.push_back(left);
					theCubesFinal.push_back(right);
				}
	
			}

		} // for each cubes

		theCubes = theCubesNext;

		dimension++;
		if (dimension > 2)
			dimension = 0;

		// we did not process any samples - this should not
		// happen, maybe the samples are out of bounds?
		// exit to avoid infinite loop.
		if (size == samples.size()) break;


	} // while samples > 0

	// we have some cubes left, but no samples - copy these to final
	if (theCubes.size()) {
		theCubesFinal.insert(theCubesFinal.end(), theCubes.begin(), theCubes.end());
	}

	// OUTPUT
	int doneVertices(0);
	for (auto& theCube : theCubesFinal)
	{

		// convert KDTree cube's 8 points to Cube mesh 24 points
		// front
		output->addPoint(scaled(theCube.center, theCube.coords[2], scale)); // 0
		output->addPoint(scaled(theCube.center, theCube.coords[1], scale)); // 1
		output->addPoint(scaled(theCube.center, theCube.coords[6], scale)); // 2
		output->addPoint(scaled(theCube.center, theCube.coords[5], scale)); // 3
		// back
		output->addPoint(scaled(theCube.center, theCube.coords[3], scale)); // 4
		output->addPoint(scaled(theCube.center, theCube.coords[0], scale)); // 5
		output->addPoint(scaled(theCube.center, theCube.coords[7], scale)); // 6
		output->addPoint(scaled(theCube.center, theCube.coords[4], scale)); // 7
		// top
		output->addPoint(scaled(theCube.center, theCube.coords[0], scale)); // 8
		output->addPoint(scaled(theCube.center, theCube.coords[4], scale)); // 9
		output->addPoint(scaled(theCube.center, theCube.coords[1], scale)); // 10
		output->addPoint(scaled(theCube.center, theCube.coords[5], scale)); // 11
		// bottom
		output->addPoint(scaled(theCube.center, theCube.coords[3], scale)); // 12
		output->addPoint(scaled(theCube.center, theCube.coords[7], scale)); // 13
		output->addPoint(scaled(theCube.center, theCube.coords[2], scale)); // 14
		output->addPoint(scaled(theCube.center, theCube.coords[6], scale)); // 15
		// right
		output->addPoint(scaled(theCube.center, theCube.coords[7], scale)); // 16
		output->addPoint(scaled(theCube.center, theCube.coords[6], scale)); // 17
		output->addPoint(scaled(theCube.center, theCube.coords[4], scale)); // 18
		output->addPoint(scaled(theCube.center, theCube.coords[5], scale)); // 19
		// left
		output->addPoint(scaled(theCube.center, theCube.coords[3], scale)); // 20
		output->addPoint(scaled(theCube.center, theCube.coords[2], scale)); // 21
		output->addPoint(scaled(theCube.center, theCube.coords[0], scale)); // 22
		output->addPoint(scaled(theCube.center, theCube.coords[1], scale)); // 23

		for (size_t i = 0; i < theCubeNormals.size(); i++) {
			output->setNormal(theCubeNormals[i], i + doneVertices);
		}
		
		for (size_t i = 0; i < theCubeVertices.size(); i++) {
			if (i % 3 == 0) {
				output->addTriangle(
					theCubeVertices[i]   + doneVertices,
					theCubeVertices[i+1] + doneVertices,
					theCubeVertices[i+2] + doneVertices
				);
			}
		}
	
		//setPointTexCoords(output, theCubeTexture.data(), theCubeNumPts);

		// shift next indices by 24
		doneVertices += 24;	

	} // for each final cube
}

void
ShapeGenerator::outputVoronoi(const OP_CHOPInput*		input, 
															float									scale,
															SOP_Output*						output) const
{
	// Set up constants for the container geometry
	const double x_min = -1, x_max = 1;
	const double y_min = -1, y_max = 1;
	const double z_min = -1, z_max = 1;
	// const double cvol = (x_max - x_min) * (y_max - y_min) * (x_max - x_min);
	
	// optimal container division is 5 cells per block
	// get fast ceiling of numSamples/5
	const int division = input->numSamples / 5 + (input->numSamples % 5 != 0);
	

	// remember how much vertices we already processed;
	// we are combining all cells to 1 shape, so we need
	// to shift the indices and normals number
	int doneVertices(0);
	int doneIndices(0);
	int currentIndices(0);


	// create the Voronoi container
	container con(
		x_min, x_max, 
		y_min, y_max, 
		z_min, z_max, 
		division, division, division,
		false, false, false, 
		8
	);


	// read voronoi points positions from input chop
	for (int i = 0; i < input->numSamples; i++) {
		con.put(i,
			input->getChannelData(0)[i],
			input->getChannelData(1)[i],
			input->getChannelData(2)[i]
		);
	}
	
	voronoicell c; 
	c_loop_all cl(con);
	if (cl.start()) {
		do {
			if (con.compute_cell(c, cl)) {
				
				// get the position of the cell central point
				double x, y, z;
				cl.pos(x, y, z);

				// -------------------------------------------------------------------------
				// output vertices
				// TODO: we have to update the Voronoi cell output so one vertex will not
				// be shared between different cell walls. Then we will be able to create
				// normals the way TD wants it - per each vertex.
				// -------------------------------------------------------------------------
				double* ptsp = c.pts;
				for (int i = 0; i < c.p; i++, ptsp += 3) {
					
					// 1:1 scale
					if (scale >= 1.0) {
						Position thePoint(
							x + *ptsp * 0.5,
							y + ptsp[1] * 0.5,
							z + ptsp[2] * 0.5
						);
						output->addPoint(thePoint);
					}
					// scale the cell towards the cell center point
					else {
						Position thePoint(
							x * (1.0 - scale) + (x + *ptsp * 0.5) * scale,
							y * (1.0 - scale) + (y + ptsp[1] * 0.5) * scale,
							z * (1.0 - scale) + (z + ptsp[2] * 0.5) * scale
						);
						output->addPoint(thePoint);
					}
				}

				double* ptsp1 = c.pts;
				for (int i = 0; i < c.p; i++, ptsp1 += 3) {
					Vector theVector(
						*ptsp1,
						ptsp1[1],
						ptsp1[2]
					);

					theVector.normalize();
					output->setNormal(theVector, i + doneVertices);
				}


				// -------------------------------------------------------------------------
				// output indices
				// -------------------------------------------------------------------------
				// c.pts : the position vectors x_0, x_1, ..., x_{ p - 1 } of the polyhedron vertices.
				// c.nu  : the number of other vertices to which each is connected.
				// c.ed  : table of edges and relations. For the i-th vertex, ed[i] has 2n_i + 1 elements:
				//		A. The first n_i elements are the edges e(j, i), where e(j, i) is the j-th neighbor of
				//       vertex i. The edges are ordered according to a right - hand rule with respect to an
				//       outward - pointing normal. 
				//    B. The next n_i elements are the relations l(j, i) which satisfy
				//       the property e(l(j, i), e(j, i)) = i. 
				//    C. The final element of the ed[i] list is a back pointer used in memory allocation.
				// -------------------------------------------------------------------------
				// for vertices
				for (int i = 1; i < c.p; i++) 
					// for each of vertice connections
					for (int j = 0; j < c.nu[i]; j++) {
						int k, l, m, n;
						// current (A) edge vertex
						k = c.ed[i][j];
						// if connection not processed
						if (k >= 0) {
							// invert it (which means it's scanned)
							c.ed[i][j] = -1 - k;
							// get index of next connection (B)
							l = c.cycle_up(c.ed[i][c.nu[i] + j], k);
							// get next connected vertice and mark as processed 
							m = c.ed[k][l]; c.ed[k][l] = -1 - m;
							// repeat untill we get back to first vertex
							while (m != i) {
								n = c.cycle_up(c.ed[k][c.nu[k] + l], m);

								output->addTriangle(
									i + doneVertices,
									k + doneVertices,
									m + doneVertices
								);
								k = m; l = n;
								m = c.ed[k][l]; c.ed[k][l] = -1 - m;
								currentIndices++;
							}
						}
					}

			
				// -------------------------------------------------------------------------
				// output vertice normals
				// -------------------------------------------------------------------------
				// std::vector<int> faces;
				// c.face_orders(faces);
				// get normals for each face
				// std::vector<double> norm;
				// c.normals(norm);
				
				//for (int i = 0; i < c.p; i++) {
				//	Vector theVector(
				//		norm[3 * i],
					//	norm[3 * i + 1],
					//	norm[3 * i + 2]
				//	);
				//	output->setNormal(theVector, i + doneVertices);
				//}


				
				// remember how much vertices we already processed
				// we are combining all cells to 1 shape, so we need
				// to shift the indices and normal numbers
				doneVertices += c.p;
				doneIndices += currentIndices;
				currentIndices = 0;


			} // if compute_cell

		} while (cl.inc());

	} // if cl.start()...
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
