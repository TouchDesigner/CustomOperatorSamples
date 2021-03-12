#pragma once

class OP_Inputs;
class OP_ParameterManager;

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char NumlevelsName[] = "Numlevels";
constexpr static char NumlevelsLabel[] = "Num Levels";

constexpr static char PyramidscaleName[] = "Pyramidscale";
constexpr static char PyramidscaleLabel[] = "Pyramid Scale";

constexpr static char FastpyramidsName[] = "Fastpyramids";
constexpr static char FastpyramidsLabel[] = "Fast Pyramids";

constexpr static char WindowsizeName[] = "Windowsize";
constexpr static char WindowsizeLabel[] = "Window Size";

constexpr static char IterationsName[] = "Iterations";
constexpr static char IterationsLabel[] = "Iterations";

constexpr static char PolynName[] = "Polyn";
constexpr static char PolynLabel[] = "Poly N";

constexpr static char PolysigmaName[] = "Polysigma";
constexpr static char PolysigmaLabel[] = "Poly Sigma";

constexpr static char UsegaussianfilterName[] = "Usegaussianfilter";
constexpr static char UsegaussianfilterLabel[] = "Use Gaussian Filter";


#pragma endregion

#pragma region Menus
#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(OP_ParameterManager*);

	// Num Levels
	static int		evalNumlevels(const OP_Inputs* input);

	// Pyramid Scale
	static double		evalPyramidscale(const OP_Inputs* input);

	// Fast Pyramids
	static bool		evalFastpyramids(const OP_Inputs* input);

	// Window Size
	static int		evalWindowsize(const OP_Inputs* input);

	// Iterations
	static int		evalIterations(const OP_Inputs* input);

	// Poly N
	static int		evalPolyn(const OP_Inputs* input);

	// Poly Sigma
	static double		evalPolysigma(const OP_Inputs* input);

	// Use Gaussian Filter
	static bool		evalUsegaussianfilter(const OP_Inputs* input);


};
#pragma endregion