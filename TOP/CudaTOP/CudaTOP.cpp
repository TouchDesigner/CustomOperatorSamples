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

#include "CudaTOP.h"

#include <assert.h>
#include <cstdio>
#include "cuda_runtime.h"

// These functions are basic C function, which the DLL loader can find
// much easier than finding a C++ Class.
// The DLLEXPORT prefix is needed so the compile exports these functions from the .dll
// you are creating
extern "C"
{
DLLEXPORT
void
FillTOPPluginInfo(TOP_PluginInfo *info)
{
	// This must always be set to this constant
	info->apiVersion = TOPCPlusPlusAPIVersion;

	// Change this to change the executeMode behavior of this plugin.
	info->executeMode = TOP_ExecuteMode::CUDA;

	// The opType is the unique name for this TOP. It must start with a 
	// capital A-Z character, and all the following characters must lower case
	// or numbers (a-z, 0-9)
	info->customOPInfo.opType->setString("Cudasample");

	// The opLabel is the text that will show up in the OP Create Dialog
	info->customOPInfo.opLabel->setString("CUDA Sample");

	// Will be turned into a 3 letter icon on the nodes
	info->customOPInfo.opIcon->setString("CDA");

	// Information about the author of this OP
	info->customOPInfo.authorName->setString("Author Name");
	info->customOPInfo.authorEmail->setString("email@email.com");

	// This TOP works with 0 or 1 inputs connected
	info->customOPInfo.minInputs = 0;
	info->customOPInfo.maxInputs = 1;
}

DLLEXPORT
TOP_CPlusPlusBase*
CreateTOPInstance(const OP_NodeInfo* info, TOP_Context *context)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per TOP that is using the .dll

	// Note we can't do any OpenGL work during instantiation

	return new CudaTOP(info, context);
}

DLLEXPORT
void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context *context)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the TOP using that instance is deleted, or
	// if the TOP loads a different DLL

	delete (CudaTOP*)instance;
}

};


CudaTOP::CudaTOP(const OP_NodeInfo* info, TOP_Context *context) :
	myNodeInfo(info), myExecuteCount(0),
	myError(nullptr),
	myInputSurface(0),
	myContext(context)
{
	myOutputSurfaces.fill(0);
	cudaStreamCreate(&myStream);
}

CudaTOP::~CudaTOP()
{
	if (myInputSurface)
		cudaDestroySurfaceObject(myInputSurface);
	for (auto o : myOutputSurfaces)
	{ 
		if (o)
			cudaDestroySurfaceObject(o);
	}
	cudaStreamDestroy(myStream);
}

void
CudaTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs *inputs, void* reserved)
{
	// Setting cookEveryFrameIfAsked to true causes the TOP to cook every frame
	// but only if something asks for it's output.
	ginfo->cookEveryFrameIfAsked = true;
}

extern cudaError_t doCUDAOperation(int width, int height, int depth, OP_TexDim dim, float4 color, cudaSurfaceObject_t input, cudaSurfaceObject_t output);

static void
setupCudaSurface(cudaSurfaceObject_t* surface, cudaArray_t array)
{
	if (*surface)
	{
		cudaResourceDesc desc;
		cudaGetSurfaceObjectResourceDesc(&desc, *surface);
		if (desc.resType != cudaResourceTypeArray ||
			desc.res.array.array != array)
		{
			cudaDestroySurfaceObject(*surface);
			*surface = 0;
		}
	}

	if (!*surface)
	{
		cudaResourceDesc desc;
		desc.resType = cudaResourceTypeArray;
		desc.res.array.array = array;
		cudaCreateSurfaceObject(surface, &desc);
	}
}

void
CudaTOP::execute(TOP_Output* output, const OP_Inputs* inputs, void* reserved)
{
	myError = nullptr;
	myExecuteCount++;

	TOP_CUDAOutputInfo info;
	info.textureDesc.width = 256;
	info.textureDesc.height = 256;
	info.textureDesc.texDim = OP_TexDim::e2D;
	info.textureDesc.pixelFormat = OP_PixelFormat::BGRA8Fixed;
	info.stream = myStream;

	float ratio = static_cast<float>(info.textureDesc.height) / static_cast<float>(info.textureDesc.width);

	const OP_CUDAArrayInfo* inputArray = nullptr;
	if (inputs->getNumInputs() > 0)
	{
		const OP_TOPInput* topInput = inputs->getInputTOP(0);

		// Make our output texture match our input texture.
		info.textureDesc = topInput->textureDesc;

		if (topInput->textureDesc.pixelFormat != OP_PixelFormat::BGRA8Fixed)
		{
			myError = "CUDA Kernel is currently only written to handle 8-bit BGRA input textures.";
			return;
		}

		OP_CUDAAcquireInfo acquireInfo;
		
		acquireInfo.stream = myStream;
		inputArray = topInput->getCUDAArray(acquireInfo, nullptr);
	}

	// Primary output will be 
	const OP_CUDAArrayInfo* outputInfo = output->createCUDAArray(info, nullptr);
	if (!outputInfo)
		return;

	// Output to a second color buffer, with a different resolution. Use a Render Select TOP
	// to get this output.
	TOP_CUDAOutputInfo auxInfo;
	auxInfo.textureDesc.pixelFormat = OP_PixelFormat::BGRA8Fixed;
	auxInfo.textureDesc.width = 1280;
	auxInfo.textureDesc.height = 720;
	auxInfo.textureDesc.texDim = OP_TexDim::e2D;
	auxInfo.colorBufferIndex = 1;
	auxInfo.stream = myStream;
	const OP_CUDAArrayInfo* auxOutputInfo = output->createCUDAArray(auxInfo, nullptr);
	if (!auxOutputInfo)
		return;

	// All calls to the 'inputs' need to be made before beginCUDAOperations() is called
	double color1[3];
	inputs->getParDouble3("Color1", color1[0], color1[1], color1[2]);
	double color2[3];
	inputs->getParDouble3("Color2", color2[0], color2[1], color2[2]);

	// Now that we have gotten all of the pointers to the OP_CUDAArrayInfos that we may want, we can tell the context
	// that we are going to start doing CUDA operations. This will cause the cudaArray members of the OP_CUDAArrayInfo
	// to get filled in with valid addresses.
	if (!myContext->beginCUDAOperations(nullptr))
		return;

	setupCudaSurface(&myOutputSurfaces[0], outputInfo->cudaArray);
	if (inputArray)
		setupCudaSurface(&myInputSurface, inputArray->cudaArray);
		
	float4 c;
	c.x = (float)color1[0];
	c.y = (float)color1[1];
	c.z = (float)color1[2];
	c.w = 1.0f;

	doCUDAOperation(info.textureDesc.width, info.textureDesc.height, info.textureDesc.depth, info.textureDesc.texDim, c, myInputSurface, myOutputSurfaces[info.colorBufferIndex]);

	setupCudaSurface(&myOutputSurfaces[auxInfo.colorBufferIndex], auxOutputInfo->cudaArray);
		
	c.x = (float)color2[0];
	c.y = (float)color2[1];
	c.z = (float)color2[2];
	c.w = 1.0f;

	doCUDAOperation(auxInfo.textureDesc.width, auxInfo.textureDesc.height, auxInfo.textureDesc.depth, auxInfo.textureDesc.texDim, c, 0, myOutputSurfaces[auxInfo.colorBufferIndex]);

	myContext->endCUDAOperations(nullptr);
}

int32_t
CudaTOP::getNumInfoCHOPChans(void* reserved)
{
	// We return the number of channel we want to output to any Info CHOP
	// connected to the TOP. In this example we are just going to send one channel.
	return 1;
}

void
CudaTOP::getInfoCHOPChan(int32_t index,
						OP_InfoCHOPChan* chan,
						void* reserved)
{
	// This function will be called once for each channel we said we'd want to return
	// In this example it'll only be called once.

	if (index == 0)
	{
		chan->name->setString("executeCount");
		chan->value = (float)myExecuteCount;
	}
}

bool		
CudaTOP::getInfoDATSize(OP_InfoDATSize* infoSize, void* reserved)
{
	infoSize->rows = 1;
	infoSize->cols = 2;
	// Setting this to false means we'll be assigning values to the table
	// one row at a time. True means we'll do it one column at a time.
	infoSize->byColumn = false;
	return true;
}

void
CudaTOP::getInfoDATEntries(int32_t index,
										int32_t nEntries,
										OP_InfoDATEntries* entries,
										void* reserved)
{
	char tempBuffer[4096];

	if (index == 0)
	{
		// Set the value for the first column
#ifdef _WIN32
		strcpy_s(tempBuffer, "executeCount");
#else // macOS
		strlcpy(tempBuffer, "executeCount", sizeof(tempBuffer));
#endif
		entries->values[0]->setString(tempBuffer);

		// Set the value for the second column
#ifdef _WIN32
		sprintf_s(tempBuffer, "%d", myExecuteCount);
#else // macOS
		snprintf(tempBuffer, sizeof(tempBuffer), "%d", myExecuteCount);
#endif
		entries->values[1]->setString(tempBuffer);
	}
}

void
CudaTOP::getErrorString(OP_String *error, void* reserved)
{
	error->setString(myError);
}

void
CudaTOP::setupParameters(OP_ParameterManager* manager, void* reserved)
{
	// color 1
	{
		OP_NumericParameter	np;

		np.name = "Color1";
		np.label = "Color 1";

		np.defaultValues[0] = 1.0;
		np.defaultValues[1] = 0.5;
		np.defaultValues[2] = 0.8;

		for (int i=0; i<3; i++)
		{
			np.minValues[i] = 0.0;
			np.maxValues[i] = 1.0;
			np.minSliders[i] = 0.0;
			np.maxSliders[i] = 1.0;
			np.clampMins[i] = true;
			np.clampMaxes[i] = true;
		}
		
		OP_ParAppendResult res = manager->appendRGB(np);
		assert(res == OP_ParAppendResult::Success);
	}
	
	// color 1
	{
		OP_NumericParameter	np;

		np.name = "Color2";
		np.label = "Color 2";

		np.defaultValues[0] = 0.0;
		np.defaultValues[1] = 0.5;
		np.defaultValues[2] = 1.0;

		for (int i=0; i<3; i++)
		{
			np.minValues[i] = 0.0;
			np.maxValues[i] = 1.0;
			np.minSliders[i] = 0.0;
			np.maxSliders[i] = 1.0;
			np.clampMins[i] = true;
			np.clampMaxes[i] = true;
		}
		
		OP_ParAppendResult res = manager->appendRGB(np);
		assert(res == OP_ParAppendResult::Success);
	}

	// pulse
	{
		OP_NumericParameter	np;

		np.name = "Reset";
		np.label = "Reset";
		
		OP_ParAppendResult res = manager->appendPulse(np);
		assert(res == OP_ParAppendResult::Success);
	}
}

void
CudaTOP::pulsePressed(const char* name, void* reserved)
{
	if (!strcmp(name, "Reset"))
	{
		// Do something to reset here
	}
}
