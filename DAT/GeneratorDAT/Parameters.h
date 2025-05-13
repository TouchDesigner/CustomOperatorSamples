#pragma once

#include<string>
#include "CPlusPlus_Common.h"

#pragma region ParNames and ParLabels

// Names of the parameters

constexpr static char SeedName[] = "Seed";
constexpr static char SeedLabel[] = "Seed";

constexpr static char RowsName[] = "Rows";
constexpr static char RowsLabel[] = "Rows";

constexpr static char ColumnsName[] = "Columns";
constexpr static char ColumnsLabel[] = "Columns";

constexpr static char LengthName[] = "Length";
constexpr static char LengthLabel[] = "Length";


#pragma endregion

#pragma region Menus
#pragma endregion


#pragma region Parameters
class Parameters
{
public:
	static void		setup(TD::OP_ParameterManager*);

	// Seed
	static double		evalSeed(const TD::OP_Inputs* inputs);

	// Rows
	static int		evalRows(const TD::OP_Inputs* inputs);

	// Columns
	static int		evalColumns(const TD::OP_Inputs* inputs);

	// Length
	static int		evalLength(const TD::OP_Inputs* inputs);


};
#pragma endregion