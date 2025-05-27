#pragma once

#include<string>

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char OperationName[] = "Operation";
constexpr static char OperationLabel[] = "Operation";

constexpr static char ResetName[] = "Reset";
constexpr static char ResetLabel[] = "Reset";


#pragma endregion

#pragma region Menus
enum class OperationMenuItems
{
	Max,
	Min,
	Average
};

#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Operation
	static OperationMenuItems		evalOperation(const TD::OP_Inputs* inputs);

	// Reset
	static int		evalReset(const TD::OP_Inputs* inputs);


};
#pragma endregion