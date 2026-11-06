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

#include "BasicGeneratorCHOP.h"
#include "Parameters.h"

#include <string>
#include <array>
#include <cassert>
#include <cmath>

// ----------------------------------------------------------------------------
// Python
//
// When installed as a Custom Operator, TouchDesigner gives each node a Python
// class with the methods and attributes listed below, e.g. in a Textport:
//
//   n = op('basicgenerator1')
//   n.Offset = 0.5        # added to every sample
//   n.ExecuteCount        # how many times the node has cooked
//   n.Reset()             # sets Offset back to 0
//
// Each function gets the node's Python object (a PY_Struct) as 'self'. Its
// context finds the C++ instance behind it with getNodeInstance(), which
// returns nullptr (with a Python error set) if the node has been deleted while
// the Python object is still around.
// ----------------------------------------------------------------------------

static PyObject*
pyReset(PyObject* self, PyObject*)
{
	PY_Struct* me = (PY_Struct*)self;

	PY_GetInfo info;
	// Resetting doesn't depend on the node's current output, so don't cook it first.
	info.autoCook = false;
	BasicGeneratorCHOP* inst = (BasicGeneratorCHOP*)me->context->getNodeInstance(info);
	if (!inst)
		return nullptr;

	inst->setOffset(0.0);
	// The output changes, so make the node cook again when it's next needed.
	me->context->makeNodeDirty();
	Py_RETURN_NONE;
}

static PyMethodDef methods[] =
{
	{"Reset", pyReset, METH_NOARGS, "Sets Offset back to 0."},
	{nullptr, nullptr, 0, nullptr}	// end of the list
};

static PyObject*
pyGetOffset(PyObject* self, void*)
{
	PY_Struct* me = (PY_Struct*)self;

	PY_GetInfo info;
	// The offset is a setting held by the instance, not a result of cooking.
	info.autoCook = false;
	BasicGeneratorCHOP* inst = (BasicGeneratorCHOP*)me->context->getNodeInstance(info);
	if (!inst)
		return nullptr;

	return PyFloat_FromDouble(inst->getOffset());
}

static int
pySetOffset(PyObject* self, PyObject* value, void*)
{
	if (!value)
	{
		PyErr_SetString(PyExc_AttributeError, "Offset can't be deleted");
		return -1;
	}

	// Accepts a float, an int, or anything with __float__; otherwise sets a TypeError.
	double v = PyFloat_AsDouble(value);
	if (v == -1.0 && PyErr_Occurred())
		return -1;

	PY_Struct* me = (PY_Struct*)self;

	PY_GetInfo info;
	info.autoCook = false;
	BasicGeneratorCHOP* inst = (BasicGeneratorCHOP*)me->context->getNodeInstance(info);
	if (!inst)
		return -1;

	inst->setOffset(v);
	me->context->makeNodeDirty();
	return 0;
}

static PyObject*
pyGetExecuteCount(PyObject* self, void*)
{
	PY_Struct* me = (PY_Struct*)self;

	PY_GetInfo info;
	// Cook the node first if it needs it, so the count is up to date.
	info.autoCook = true;
	BasicGeneratorCHOP* inst = (BasicGeneratorCHOP*)me->context->getNodeInstance(info);
	if (!inst)
		return nullptr;

	return PyLong_FromLong(inst->getExecuteCount());
}

static PyGetSetDef getSets[] =
{
	{"Offset", pyGetOffset, pySetOffset, "Get or set the value added to every sample.", nullptr},
	// No setter, so this one is read-only.
	{"ExecuteCount", pyGetExecuteCount, nullptr, "Number of times the node has cooked.", nullptr},
	{nullptr, nullptr, nullptr, nullptr, nullptr}	// end of the list
};

// Contents of the callbacks DAT that TouchDesigner creates for the node.
// execute() calls getScale() on every cook.
static const char* PythonCallbacksDATStubs =
"# op    - The Basic Generator CHOP doing the callback.\n"
"# scale - The scale it is about to use (1.0 when Apply Scale is off).\n"
"#\n"
"# Return the scale to use instead, e.g. scale * 2.\n"
"def getScale(op, scale):\n"
"\treturn scale\n";

// These functions are basic C function, which the DLL loader can find
// much easier than finding a C++ Class.
// The DLLEXPORT prefix is needed so the compile exports these functions from the .dll
// you are creating
extern "C"
{

DLLEXPORT
void
FillCHOPPluginInfo(CHOP_PluginInfo *info)
{
	// For more information on CHOP_PluginInfo see CHOP_CPlusPlusBase.h

	// Always set this to CHOPCPlusPlusAPIVersion.
	info->apiVersion = CHOPCPlusPlusAPIVersion;

	// For more information on OP_CustomOPInfo see CPlusPlus_Common.h
	OP_CustomOPInfo& customInfo = info->customOPInfo;

	// Unique name of the node which starts with an upper case letter, followed by lower case letters or numbers
	customInfo.opType->setString("Basicgenerator");
	// English readable name
	customInfo.opLabel->setString("Basic Generator");
	// Information of the author of the node
	customInfo.authorName->setString("Author Name");
	customInfo.authorEmail->setString("email@email");

	// This CHOP takes no inputs
	customInfo.minInputs = 0;
	customInfo.maxInputs = 0;

	// Python: the version the plugin was built against (TouchDesigner checks it
	// matches its own), the methods and attributes of the node's Python class,
	// and the contents of its callbacks DAT.
	customInfo.pythonVersion->setString(PY_VERSION);
	customInfo.pythonMethods = methods;
	customInfo.pythonGetSets = getSets;
	customInfo.pythonCallbacksDAT = PythonCallbacksDATStubs;
}

DLLEXPORT
CHOP_CPlusPlusBase*
CreateCHOPInstance(const OP_NodeInfo* info)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per CHOP that is using the .dll
	return new BasicGeneratorCHOP(info);
}

DLLEXPORT
void
DestroyCHOPInstance(CHOP_CPlusPlusBase* instance)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the CHOP using that instance is deleted, or
	// if the CHOP loads a different DLL
	delete (BasicGeneratorCHOP*)instance;
}

};


BasicGeneratorCHOP::BasicGeneratorCHOP(const OP_NodeInfo* info) :
	myNodeInfo(info)
{

}

BasicGeneratorCHOP::~BasicGeneratorCHOP()
{

}

void
BasicGeneratorCHOP::getGeneralInfo(CHOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
	// This chop doesn't need to cook every frame
	ginfo->cookEveryFrameIfAsked = false;

	// Note: To disable timeslicing you'll need to turn this off, as well as ensure that
	// getOutputInfo() returns true, and likely also set the info->numSamples to how many
	// samples you want to generate for this CHOP. Otherwise it'll take on length of the
	// input CHOP, which may be timesliced.
	ginfo->timeslice = false;
}

bool
BasicGeneratorCHOP::getOutputInfo(CHOP_OutputInfo* info, const OP_Inputs* inputs, void*)
{
	info->numChannels = inputs->getParInt("Numberofchannels");
	info->numSamples = inputs->getParInt("Length");
	info->startIndex = 0;
	return true;
}

void
BasicGeneratorCHOP::getChannelName(int32_t index, OP_String *name, const OP_Inputs*, void*)
{
	std::string channelName = "chan" + std::to_string(index);
	name->setString(channelName.c_str());
}

void
BasicGeneratorCHOP::execute(CHOP_Output* output,
							  const OP_Inputs* inputs,
							  void*)
{
	myExecuteCount++;

	// Get all Parameters
	bool	applyScale = myParms.evalApplyscale(inputs);
	double	scale = applyScale ? myParms.evalScale(inputs) : 1.0;
	OperationMenuItems		operation = myParms.evalOperation(inputs);

	inputs->enablePar(ScaleName, applyScale);

	// Let the callbacks DAT adjust the scale: getScale(op, scale).
	// createArgumentsTuple() reserves the first item for 'op'; we add one more.
	PyObject* args = myNodeInfo->context->createArgumentsTuple(1, nullptr);
	if (args)
	{
		PyTuple_SET_ITEM(args, 1, PyFloat_FromDouble(scale));	// steals the reference

		PyObject* result = myNodeInfo->context->callPythonCallback("getScale", args, nullptr, nullptr);
		// callPythonCallback() doesn't take ownership of the arguments.
		Py_DECREF(args);

		// We own the result. It's nullptr if there's no callbacks DAT or no
		// getScale() in it (e.g. when loaded into a CPlusPlus CHOP).
		if (result)
		{
			if (PyFloat_Check(result) || PyLong_Check(result))
				scale = PyFloat_AsDouble(result);
			Py_DECREF(result);
		}
	}

	double	offset = myOffset;

	// Get length and channels from the output since they first need to be set to the current size
	int		length = output->numSamples;
	int		channels = output->numChannels;

	// Calculate scale*(channel operation sample) + offset
	double	curValue;
	for (int i = 0; i < channels; ++i) 
	{
		for (int j = 0; j < length; ++j) 
		{
			switch (operation) 
			{
				case OperationMenuItems::Add:
				{
					curValue = (double)i + j;
					break;
				}
				case OperationMenuItems::Multiply:
				{
					curValue = (double)i * j;
					break;
				}
				case OperationMenuItems::Power:
				default:
				{
					curValue = std::pow((double)i, (double)j);
					break;
				}
			}

			curValue = curValue * scale + offset;

			output->channels[i][j] = (float)curValue;
		}
	}
}

void
BasicGeneratorCHOP::setupParameters(OP_ParameterManager* manager, void*)
{
	myParms.setup(manager);
}
