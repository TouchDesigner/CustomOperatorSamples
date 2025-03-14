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

#include "GeneratorDAT.h"
#include "Parameters.h"

#include <string>
#include <random>
#include <array>

static const char ALPHANUM[] =
"0123456789"
"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
"abcdefghijklmnopqrstuvwxyz";

// These functions are basic C function, which the DLL loader can find
// much easier than finding a C++ Class.
// The DLLEXPORT prefix is needed so the compile exports these functions from the .dll
// you are creating
extern "C"
{

DLLEXPORT
void
FillDATPluginInfo(DAT_PluginInfo *info)
{
	// For more information on CHOP_PluginInfo see CHOP_CPlusPlusBase.h

	// Always set this to CHOPCPlusPlusAPIVersion.
	info->apiVersion = DATCPlusPlusAPIVersion;

	// For more information on OP_CustomOPInfo see CPlusPlus_Common.h
	OP_CustomOPInfo& customInfo = info->customOPInfo;

	// Unique name of the node which starts with an upper case letter, followed by lower case letters or numbers
	customInfo.opType->setString("Generator");
	// English readable name
	customInfo.opLabel->setString("Generator");
	// Information of the author of the node
	customInfo.authorName->setString("Author Name");
	customInfo.authorEmail->setString("email@email");

	// This Dat takes no input
	customInfo.minInputs = 0;
	customInfo.maxInputs = 0;
}

DLLEXPORT
DAT_CPlusPlusBase*
CreateDATInstance(const OP_NodeInfo* info)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per DAT that is using the .dll
	return new GeneratorDAT(info);
}

DLLEXPORT
void
DestroyDATInstance(DAT_CPlusPlusBase* instance)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the DAT using that instance is deleted, or
	// if the DAT loads a different DLL
	delete (GeneratorDAT*)instance;
}

};

GeneratorDAT::GeneratorDAT(const OP_NodeInfo* info) :
	myRNG{}
{
}

GeneratorDAT::~GeneratorDAT()
{
}

void
GeneratorDAT::getGeneralInfo(DAT_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	// This will cause the node to not cook every frame
	ginfo->cookEveryFrameIfAsked = false;
}

void
GeneratorDAT::execute(DAT_Output* output, const OP_Inputs* inputs, void*)
{
	double mySeed = myParms.evalSeed(inputs);
	unsigned int* tmp = reinterpret_cast<unsigned int*>(&mySeed);
	myRNG.seed(*tmp);

	output->setOutputDataType(DAT_OutDataType::Table);
	output->setTableSize(myParms.evalRows(inputs), myParms.evalColumns(inputs));

	fillTable(inputs, output);
}

void
GeneratorDAT::setupParameters(OP_ParameterManager* manager, void*)
{
	myParms.setup(manager);
}

void
GeneratorDAT::fillTable(const OP_Inputs* inputs, DAT_Output* out)
{

	for (int i = 0; i < myParms.evalRows(inputs); ++i)
	{
		for (int j = 0; j < myParms.evalColumns(inputs); ++j)
		{
			out->setCellString(i, j, generateString(inputs).c_str());
		}
	}
}

std::string
GeneratorDAT::generateString(const OP_Inputs* inputs)
{
	std::uniform_int_distribution<> dis(0, sizeof(ALPHANUM) - 2);
	std::string ret{};
	for (int i = 0; i <myParms.evalLength(inputs); ++i)
	{
		ret.push_back(ALPHANUM[dis(myRNG)]);
	}

	return ret;
}
