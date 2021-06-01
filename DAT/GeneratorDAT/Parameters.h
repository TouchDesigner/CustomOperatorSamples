#pragma once

class OP_Inputs;
class OP_ParameterManager;

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
	static void		setup(OP_ParameterManager*);

	// Seed
	static double		evalSeed(const OP_Inputs* input);

	// Rows
	static int		evalRows(const OP_Inputs* input);

	// Columns
	static int		evalColumns(const OP_Inputs* input);

	// Length
	static int		evalLength(const OP_Inputs* input);


};
#pragma endregion