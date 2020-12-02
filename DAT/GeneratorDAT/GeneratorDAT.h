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

#ifndef __GeneratorDAT__
#define __GeneratorDAT__

#include "DAT_CPlusPlusBase.h"

#include <random>

/*
This example implements a DAT that generates random text. This DAT is a generator so it takes
no inputs.

It takes the following parameters:
	- Seed:	A number used to initialize the pseudorandom generator.
	- Rows:	The number of rows to output.
	- Columns:	The number of columns to output.
	- Length: The length of the text in each cell of the table.
*/

// Check methods [getNumInfoCHOPChans, getInfoCHOPChan, getInfoDATSize, getInfoDATEntries]
// if you want to output values to the Info CHOP/DAT

// To get more help about these functions, look at SOP_CPlusPlusBase.h

class GeneratorDAT : public DAT_CPlusPlusBase
{
public:
	GeneratorDAT(const OP_NodeInfo*);
	virtual ~GeneratorDAT();

	virtual void		getGeneralInfo(DAT_GeneralInfo*, const OP_Inputs*, void* reserved) override;

	virtual void		execute(DAT_Output*, const OP_Inputs*, void*) override;

	virtual void		setupParameters(OP_ParameterManager*, void*) override;

private:
	void			handleParameters(const OP_Inputs*);

	void			fillTable(DAT_Output*);

	std::string		generateString();

	std::mt19937	myRNG;

	// Parameters
	double		mySeed;
	int			myRowsN;
	int			myColsN;
	int			myCharN;
};

#endif
