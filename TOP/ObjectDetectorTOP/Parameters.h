#pragma once

class OP_Inputs;
class OP_ParameterManager;

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char ClassifierName[] = "Classifier";
constexpr static char ClassifierLabel[] = "Classifier";

constexpr static char ScalefactorName[] = "Scalefactor";
constexpr static char ScalefactorLabel[] = "Scale Factor";

constexpr static char MinneighborsName[] = "Minneighbors";
constexpr static char MinneighborsLabel[] = "Min Neighbors";

constexpr static char LimitobjectsizeName[] = "Limitobjectsize";
constexpr static char LimitobjectsizeLabel[] = "Limit Object Size";

constexpr static char MinobjectwidthName[] = "Minobjectwidth";
constexpr static char MinobjectwidthLabel[] = "Min Object Width";

constexpr static char MinobjectheightName[] = "Minobjectheight";
constexpr static char MinobjectheightLabel[] = "Min Object Height";

constexpr static char MaxobjectwidthName[] = "Maxobjectwidth";
constexpr static char MaxobjectwidthLabel[] = "Max Object Width";

constexpr static char MaxobjectheightName[] = "Maxobjectheight";
constexpr static char MaxobjectheightLabel[] = "Max Object Height";

constexpr static char DrawboundingboxName[] = "Drawboundingbox";
constexpr static char DrawboundingboxLabel[] = "Draw Bounding Box";

constexpr static char LimitobjectsdetectedName[] = "Limitobjectsdetected";
constexpr static char LimitobjectsdetectedLabel[] = "Limit Objects Detected";

constexpr static char MaximumobjectsName[] = "Maximumobjects";
constexpr static char MaximumobjectsLabel[] = "Maximum Objects";

constexpr static char DownloadtypeName[] = "Downloadtype";
constexpr static char DownloadtypeLabel[] = "Download Type";


#pragma endregion

#pragma region Menus
enum class DownloadtypeMenuItems
{
	Delayed,
	Instant
};

#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(OP_ParameterManager*);

	// Classifier
	static std::string		evalClassifier(const OP_Inputs* input);

	// Scale Factor
	static double		evalScalefactor(const OP_Inputs* input);

	// Min Neighbors
	static int		evalMinneighbors(const OP_Inputs* input);

	// Limit Object Size
	static bool		evalLimitobjectsize(const OP_Inputs* input);

	// Min Object Width
	static double		evalMinobjectwidth(const OP_Inputs* input);

	// Min Object Height
	static double		evalMinobjectheight(const OP_Inputs* input);

	// Max Object Width
	static double		evalMaxobjectwidth(const OP_Inputs* input);

	// Max Object Height
	static double		evalMaxobjectheight(const OP_Inputs* input);

	// Draw Bounding Box
	static bool		evalDrawboundingbox(const OP_Inputs* input);

	// Limit Objects Detected
	static bool		evalLimitobjectsdetected(const OP_Inputs* input);

	// Maximum Objects
	static int		evalMaximumobjects(const OP_Inputs* input);

	// Download Type
	static DownloadtypeMenuItems		evalDownloadtype(const OP_Inputs* input);


};
#pragma endregion