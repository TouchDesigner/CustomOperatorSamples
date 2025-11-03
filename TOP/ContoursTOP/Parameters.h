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

#pragma once

namespace TD
{
	class OP_Inputs;
	class OP_ParameterManager;
}

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char ModeName[] = "Mode";
constexpr static char ModeLabel[] = "Mode";

constexpr static char MethodName[] = "Method";
constexpr static char MethodLabel[] = "Method";

constexpr static char WatershedName[] = "Watershed";
constexpr static char WatershedLabel[] = "Watershed";

constexpr static char SelectobjectName[] = "Selectobject";
constexpr static char SelectobjectLabel[] = "Select Object";

constexpr static char ObjectName[] = "Object";
constexpr static char ObjectLabel[] = "Object";

constexpr static char DownloadtypeName[] = "Downloadtype";
constexpr static char DownloadtypeLabel[] = "Download Type";

constexpr static char ChannelName[] = "Channel";
constexpr static char ChannelLabel[] = "Channel";


#pragma endregion

#pragma region Menus
enum class ModeMenuItems
{
	External,
	List,
	Ccomp,
	Tree
};

enum class MethodMenuItems
{
	None,
	Simple,
	Tcl1,
	Tckcos
};

enum class DownloadtypeMenuItems
{
	Delayed,
	Instant
};

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

	// Mode
	static ModeMenuItems		evalMode(const TD::OP_Inputs* input);

	// Method
	static MethodMenuItems		evalMethod(const TD::OP_Inputs* input);

	// Watershed
	static bool		evalWatershed(const TD::OP_Inputs* input);

	// Select Object
	static bool		evalSelectobject(const TD::OP_Inputs* input);

	// Object
	static int		evalObject(const TD::OP_Inputs* input);

	// Download Type
	static DownloadtypeMenuItems		evalDownloadtype(const TD::OP_Inputs* input);

	// Channel
	static ChannelMenuItems		evalChannel(const TD::OP_Inputs* input);


};
#pragma endregion