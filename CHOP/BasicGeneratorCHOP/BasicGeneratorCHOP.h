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

#ifndef __BasicGeneratorCHOP__
#define __BasicGeneratorCHOP__

// Python.h has to be included before any other header (see the CPython docs).
// The headers come from the TouchDesigner installation (td_add_custom_op's
// PYTHON option), so they match the Python this plugin is loaded into.
#include <Python.h>

#ifdef Py_DEBUG
	#error "Py_DEBUG is set (check for _DEBUG / /MDd); TouchDesigner's Python is a release build"
#endif

#include "CHOP_CPlusPlusBase.h"
#include "Parameters.h"

#include <atomic>

using namespace TD;

/*
This example implements a CHOP which takes the following parameters:
	- Length: The number of samples produced per channel.
	- Number of Channels: The number of channels produced.
	- Apply Scale: If On, scale values.
	- Scale: A scalar by which the output signal is scaled.
	- Operation: One of [Add, Multiply, Power] which controls which operation the output signal constitutes

The output values are: scale*(channel operation sample) + offset

This CHOP is a generator so it does not need an input and it is not time sliced.

It also shows how a Custom Operator can extend its node with Python, using the
CPython API (see the CPython docs for the Py* structures and functions used):
	- op('basicgenerator1').Offset        read/write; added to every sample
	- op('basicgenerator1').ExecuteCount  read-only; how many times the node cooked
	- op('basicgenerator1').Reset()       sets Offset back to 0
	- a callbacks DAT with a getScale(op, scale) function, which can change the
	  scale used for each cook
Note that the Python members and the callbacks DAT only exist when the plugin
is installed as a Custom Operator, not when it's loaded into a CPlusPlus CHOP.
*/

// Check methods [getNumInfoCHOPChans, getInfoCHOPChan, getInfoDATSize, getInfoDATEntries]
// if you want to output values to the Info CHOP/DAT

// To get more help about these functions, look at CHOP_CPlusPlusBase.h
class BasicGeneratorCHOP : public CHOP_CPlusPlusBase
{
public:
	BasicGeneratorCHOP(const OP_NodeInfo* info);
	virtual ~BasicGeneratorCHOP();

	virtual void		getGeneralInfo(CHOP_GeneralInfo*, const TD::OP_Inputs*, void*) override;
	virtual bool		getOutputInfo(CHOP_OutputInfo*, const TD::OP_Inputs*, void*) override;
	virtual void		getChannelName(int32_t index, OP_String *name, const TD::OP_Inputs*, void*) override;

	virtual void		execute(CHOP_Output*, const TD::OP_Inputs*, void*) override;

	virtual void		setupParameters(TD::OP_ParameterManager* manager, void*) override;

	// Used by the Python members (see BasicGeneratorCHOP.cpp). Python may call
	// these from another thread than the one cooking the node (TouchDesigner
	// builds with free-threaded Python have no GIL to serialize them), so the
	// values they share with execute() are atomic.
	double		getOffset() const { return myOffset; }
	void		setOffset(double v) { myOffset = v; }
	int32_t		getExecuteCount() const { return myExecuteCount; }

private:
	const OP_NodeInfo*		myNodeInfo;
	Parameters				myParms;

	std::atomic<double>		myOffset{0.0};
	std::atomic<int32_t>	myExecuteCount{0};
};

#endif
