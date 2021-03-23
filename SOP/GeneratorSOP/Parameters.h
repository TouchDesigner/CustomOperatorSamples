#pragma once

class OP_Inputs;
class OP_ParameterManager;

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
	static void		setup(OP_ParameterManager*);

	// Shape
	static ShapeMenuItems		evalShape(const OP_Inputs* input);

	// Color
	static Color		evalColor(const OP_Inputs* input);

	// GPU Direct
	static bool		evalGpudirect(const OP_Inputs* input);


};
#pragma endregion