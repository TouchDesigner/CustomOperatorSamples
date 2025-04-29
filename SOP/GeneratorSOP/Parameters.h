#pragma once

#include<string>

class TD::OP_Inputs;
class TD::OP_ParameterManager;

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char ShapeName[] = "Shape";
constexpr static char ShapeLabel[] = "Shape";

constexpr static char ColorName[] = "Color";
constexpr static char ColorLabel[] = "Color";

constexpr static char GpudirectName[] = "Gpudirect";
constexpr static char GpudirectLabel[] = "GPU Direct";


#pragma endregion

#pragma region Menus
enum class ShapeMenuItems
{
	Point,
	Line,
	Square,
	Cube
};

#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Shape
	static ShapeMenuItems		evalShape(const TD::OP_Inputs* inputs);

	// Color
	static TD::Color		evalColor(const TD::OP_Inputs* inputs);

	// GPU Direct
	static bool		evalGpudirect(const TD::OP_Inputs* inputs);


};
#pragma endregion