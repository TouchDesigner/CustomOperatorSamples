#pragma once

#include<string>

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char CaseName[] = "Case";
constexpr static char CaseLabel[] = "Case";

constexpr static char KeepspacesName[] = "Keepspaces";
constexpr static char KeepspacesLabel[] = "Keep Spaces";


#pragma endregion

#pragma region Menus
enum class CaseMenuItems
{
	Uppercamelcase,
	Lowercase,
	Uppercase
};

#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Case
	static CaseMenuItems		evalCase(const TD::OP_Inputs* inputs);

	// Keep Spaces
	static bool		evalKeepspaces(const TD::OP_Inputs* inputs);


};
#pragma endregion