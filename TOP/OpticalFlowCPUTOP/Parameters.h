#pragma once

#include<string>

class TD::OP_Inputs;
class TD::OP_ParameterManager;

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char NumlevelsName[] = "Numlevels";
constexpr static char NumlevelsLabel[] = "Num Levels";

constexpr static char PyramidscaleName[] = "Pyramidscale";
constexpr static char PyramidscaleLabel[] = "Pyramid Scale";

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

constexpr static char UsepreviousflowName[] = "Usepreviousflow";
constexpr static char UsepreviousflowLabel[] = "Use Previous Flow";

constexpr static char ChannelName[] = "Channel";
constexpr static char ChannelLabel[] = "Channel";


#pragma endregion

#pragma region Menus
enum class ChannelMenuItems
{
	R,
	G,
	B,
	A
};

#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Num Levels
	static int		evalNumlevels(const TD::OP_Inputs* inputs);

	// Pyramid Scale
	static double		evalPyramidscale(const TD::OP_Inputs* inputs);

	// Window Size
	static int		evalWindowsize(const TD::OP_Inputs* inputs);

	// Iterations
	static int		evalIterations(const TD::OP_Inputs* inputs);

	// Poly N
	static int		evalPolyn(const TD::OP_Inputs* inputs);

	// Poly Sigma
	static double		evalPolysigma(const TD::OP_Inputs* inputs);

	// Use Gaussian Filter
	static bool		evalUsegaussianfilter(const TD::OP_Inputs* inputs);

	// Use Previous Flow
	static bool		evalUsepreviousflow(const TD::OP_Inputs* inputs);

	// Channel
	static ChannelMenuItems		evalChannel(const TD::OP_Inputs* inputs);


};
#pragma endregion