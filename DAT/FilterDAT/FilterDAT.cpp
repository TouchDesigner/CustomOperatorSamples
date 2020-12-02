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

#include "FilterDAT.h"

#include <string>
#include <cctype>

// Names of the parameters
constexpr static char	CASE_NAME[] = "Case";
constexpr static char	WHITESPACE_NAME[] = "Whitespace";

enum class
Case
{
	UpperCamel,
	Lower,
	Upper
};

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
	customInfo.opType->setString("Filter");
	// English readable name
	customInfo.opLabel->setString("Filter");
	// Information of the author of the node
	customInfo.authorName->setString("Name");
	customInfo.authorEmail->setString("name@domain.com");

	// This Dat takes no input
	customInfo.minInputs = 1;
	customInfo.maxInputs = 1;
}

DLLEXPORT
DAT_CPlusPlusBase*
CreateDATInstance(const OP_NodeInfo* info)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per DAT that is using the .dll
	return new DATFilter(info);
}

DLLEXPORT
void
DestroyDATInstance(DAT_CPlusPlusBase* instance)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the DAT using that instance is deleted, or
	// if the DAT loads a different DLL
	delete (DATFilter*)instance;
}

};

namespace
{
	void
	toCamelCase(std::string& str, bool keepSpaces)
	{
		std::string	in{ std::move(str) };
		str.clear();

		bool		nextUpper = true;

		for (char c : in)
		{
			if (std::isspace(c))
			{
				nextUpper = true;
				if (keepSpaces)
					str.push_back(c);
			}
			else if (nextUpper)
			{
				str.push_back(std::toupper(c));
				nextUpper = false;
			}
			else
			{
				str.push_back(std::tolower(c));
			}
		}
	}

	void
	changeCase(std::string& str, bool keepSpaces, bool upper)
	{
		std::string in{ std::move(str) };
		str.clear();

		for (char c : in)
		{
			if (keepSpaces || !std::isspace(c))
			{
				str.push_back(upper ? std::toupper(c) : std::tolower(c));
			}
		}
	}

	void
	toUpperCase(std::string& str, bool keepSpaces)
	{
		changeCase(str, keepSpaces, true);
	}

	void
	toLowerCase(std::string& str, bool keepSpaces)
	{
		changeCase(str, keepSpaces, false);
	}
};

DATFilter::DATFilter(const OP_NodeInfo* info)
{
}

DATFilter::~DATFilter()
{
}

void
DATFilter::getGeneralInfo(DAT_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	// This will cause the node to not cook every frame
	ginfo->cookEveryFrameIfAsked = false;
}

void
DATFilter::execute(DAT_Output* output, const OP_Inputs* inputs, void*)
{
	const OP_DATInput* dat	= inputs->getInputDAT(0);
	if (!dat)
		return;

	handleParameters(inputs);

	output->setOutputDataType(dat->isTable ? DAT_OutDataType::Table : DAT_OutDataType::Text);
	output->setTableSize(dat->numRows, dat->numCols);

	fillTable(output, dat);
}

void 
DATFilter::setupParameters(OP_ParameterManager* manager, void*)
{
	{
		OP_StringParameter sp;
		sp.name = CASE_NAME;
		sp.label = "Case";

		sp.defaultValue = "UpperCamelCase";

		const char* names[] = { "Uppercamelcase", "Lowercase", "Uppercase" };
		const char* labels[] = { "Upper Camel Case", "Lower Case", "Upper Case" };

		OP_ParAppendResult res = manager->appendMenu(sp, 3, names, labels);
		assert(res == OP_ParAppendResult::Success);
	}

	{
		OP_NumericParameter np;
		np.name = WHITESPACE_NAME;
		np.label = "Keep Spaces";

		np.defaultValues[0] = true;

		OP_ParAppendResult res = manager->appendToggle(np);
		assert(res == OP_ParAppendResult::Success);
	}
}

void 
DATFilter::handleParameters(const OP_Inputs* in)
{
	myCase = static_cast<Case>(in->getParInt(CASE_NAME));
	myKeepSpaces = in->getParInt(WHITESPACE_NAME) ? true : false;
}

void
DATFilter::fillTable(DAT_Output* out, const OP_DATInput* in)
{
	for (int i = 0; i < in->numRows; ++i)
	{
		for (int j = 0; j < in->numCols; ++j)
		{
			std::string tmp{ in->getCell(i,j) };
			switch (myCase)
			{
			default:
			case Case::UpperCamel:
				toCamelCase(tmp, myKeepSpaces);
				break;
			case Case::Lower:
				toLowerCase(tmp, myKeepSpaces);
				break;
			case Case::Upper:
				toUpperCase(tmp, myKeepSpaces);
				break;
			}
			out->setCellString(i, j, tmp.c_str());
		}
	}
}
