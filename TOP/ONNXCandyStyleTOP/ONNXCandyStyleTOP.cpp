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

#include "ONNXCandyStyleTOP.h"

#include <assert.h>
#include <cstdio>
#include <string>
#include <cwchar>
#include "cuda_runtime.h"
#include <onnxruntime_run_options_config_keys.h>

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
	info->customOPInfo.opType->setString("Candystyle");

	// The opLabel is the text that will show up in the OP Create Dialog
	info->customOPInfo.opLabel->setString("Candy Style");

	// Will be turned into a 3 letter icon on the nodes
	info->customOPInfo.opIcon->setString("CST");

	// Information about the author of this OP
	info->customOPInfo.authorName->setString("Author Name");
	info->customOPInfo.authorEmail->setString("email@email.com");

	// This TOP works with 1 input connected
	info->customOPInfo.minInputs = 1;
	info->customOPInfo.maxInputs = 1;
}

DLLEXPORT
TOP_CPlusPlusBase*
CreateTOPInstance(const OP_NodeInfo* info, TOP_Context *context)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per TOP that is using the .dll

	// Note we can't do any OpenGL work during instantiation

	return new ONNXCandyStyleTOP(info, context);
}

DLLEXPORT
void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context *context)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the TOP using that instance is deleted, or
	// if the TOP loads a different DLL

	// We do some OpenGL teardown on destruction, so ask the TOP_Context
	// to set up our OpenGL context

	delete (ONNXCandyStyleTOP*)instance;
}

};


ONNXCandyStyleTOP::ONNXCandyStyleTOP(const OP_NodeInfo* info, TOP_Context *context) :
	myNodeInfo(info),
	myContext(context)
{
	cudaStreamCreate(&myStream);
}

ONNXCandyStyleTOP::~ONNXCandyStyleTOP()
{
	if (myORT)
	{
		if (mySourceTensor)
			myORT->ReleaseValue(mySourceTensor);
		if (myDestTensor)
			myORT->ReleaseValue(myDestTensor);
		if (myRunOptions)
			myORT->ReleaseRunOptions(myRunOptions);
		if (mySessionOptions)
			myORT->ReleaseSessionOptions(mySessionOptions);
		if (mySession)
			myORT->ReleaseSession(mySession);
		if (myEnv)
			myORT->ReleaseEnv(myEnv);
	}
	if (myStream)
		cudaStreamDestroy(myStream);
	if (myInputSurface)
		cudaDestroySurfaceObject(myInputSurface);
	if (myOutputSurface)
		cudaDestroySurfaceObject(myOutputSurface);
	if (mySourceData)
		cudaFree(mySourceData);
	if (myDestData)
		cudaFree(myDestData);
}

void
ONNXCandyStyleTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs *inputs, void* reserved)
{
}

// Declare these here so the actual implementions that are done in kernel.cu can be found.
extern cudaError_t doCopyRGBATToRGBPlanar(int width, int height, int depth, TD::OP_TexDim dim, cudaSurfaceObject_t input, float* output, cudaStream_t stream);
extern cudaError_t doCopyRGBPlanarToRGBA(int width, int height, int depth, TD::OP_TexDim dim, float* input, cudaSurfaceObject_t output, cudaStream_t stream);

static bool
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
	return *surface != 0;
}

bool
ONNXCandyStyleTOP::checkORTStatus(OrtStatus* status, std::string* err)
{
	if (status)
	{
		*err = myORT->GetErrorMessage(status);
		myORT->ReleaseStatus(status);
		return false;
	}
	return true;
}

bool
ONNXCandyStyleTOP::allocCUDAAndValue(void** cudaMemory, OrtValue** value, size_t size, const std::array<int64_t, 4>& shape)
{
	auto res = cudaMalloc(cudaMemory, size);
	if (res != cudaSuccess || !*cudaMemory)
	{
		myError = "Failed to allocate memory for CUDA source.";
		return false;
	}
	OrtMemoryInfo* memoryInfo;
	std::string extraErr;
	OrtStatus* status = myORT->CreateMemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault, &memoryInfo);
	if (!checkORTStatus(status, &extraErr) || !memoryInfo)
	{
		myError = "Failed to create OrtMemoryInfo for CUDA memory. ";
		myError += extraErr;
		return false;
	}

	status = myORT->CreateTensorWithDataAsOrtValue(memoryInfo, *cudaMemory, size, shape.data(), shape.size(),
													ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, value);
	myORT->ReleaseMemoryInfo(memoryInfo);
	if (!checkORTStatus(status, &extraErr) || !*value)
	{
		myError = "Failed to create OrtValue. ";
		myError += extraErr;
		return false;
	}
	int isTensor = 0;
	status = myORT->IsTensor(*value, &isTensor);
	if (!checkORTStatus(status, &extraErr) || !isTensor)
	{
		myORT->ReleaseValue(*value);
		myError = "Created OvtValue is invalid. ";
		myError += extraErr;
		return false;
	}
	return true;
}

void
ONNXCandyStyleTOP::execute(TOP_Output* output, const OP_Inputs* inputs, void* reserved)
{
	if (inputs->getNumInputs() < 1)
		return;

	myError.clear();
	if (!myORT)
	{
		std::string extraErr;
		myORT = OrtGetApiBase()->GetApi(ORT_API_VERSION);
		if (!myORT)
		{
			myError = "Failed to initialize ONNX runtime. Likely this plugin was compiled against headers newer than the onnx binaries included in TouchDesigner.";
			return;
		}
		OrtStatus* status = myORT->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "CandyTOP", &myEnv);
		if (!checkORTStatus(status, &extraErr) || !myEnv)
		{
			myError = "Failed to create ONNX environment. ";
			myError += extraErr;
			return;
		}
		status = myORT->CreateSessionOptions(&mySessionOptions);
		if (!checkORTStatus(status, &extraErr) || !mySessionOptions)
		{
			myError = "Failed to create session options. ";
			myError += extraErr;
			return;
		}

		OrtCUDAProviderOptions o = {};
		o.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
		o.gpu_mem_limit = SIZE_MAX;
		o.user_compute_stream = myStream;
		status = myORT->SessionOptionsAppendExecutionProvider_CUDA(mySessionOptions, &o);
		if (!checkORTStatus(status, &extraErr))
		{
			myError = "Failed to set execution provider to be CUDA. ";
			myError += extraErr;
			return;
		}

		std::string s(myNodeInfo->pluginPath);
		size_t lastSlash = s.find_last_of('/');
		if (lastSlash != std::string::npos)
			s.erase(lastSlash);
		s += "/candy.onnx";
		ORTCHAR_T fullPath[4096];
		const char* src = s.c_str();
		std::mbstate_t ps{};
		size_t retVal;
		mbsrtowcs_s(&retVal, fullPath, &src, sizeof(fullPath) - 1, &ps);
		status = myORT->CreateSession(myEnv, fullPath, mySessionOptions, &mySession);
		if (!checkORTStatus(status, &extraErr) || !mySession)
		{
			myError = "Unable to create Sessions. Ensure candy.onnx is next to the plugin's .dll. ";
			myError += extraErr;
			return;
		}
		status = myORT->SessionGetInputCount(mySession, &myInputCount);
		if (!checkORTStatus(status, &extraErr) || myInputCount != 1)
		{
			myError = "The loaded sessions must only have 1 input. ";
			myError += extraErr;
			return;
		}
		status = myORT->SessionGetOutputCount(mySession, &myOutputCount);
		if (!checkORTStatus(status, &extraErr) || myOutputCount != 1)
		{
			myError = "The loaded sessions must only have 1 output. ";
			myError += extraErr;
			return;
		}
		status = myORT->CreateRunOptions(&myRunOptions);
		if (!checkORTStatus(status, &extraErr) || !myRunOptions)
		{
			myError = "Failed to create run options. ";
			myError += extraErr;
			return;
		}
		status = myORT->AddRunConfigEntry(myRunOptions, kOrtRunOptionsConfigDisableSynchronizeExecutionProviders, "1");
		if (!checkORTStatus(status, &extraErr))
		{
			myError = "Failed to set run option value. ";
			myError += extraErr;
			return;
		}
	}

	if (!myORT || !mySession)
		return;

	const OP_TOPInput* topInput = inputs->getInputTOP(0);

	const int InputWidth = 720;
	const int InputHeight = 720;

	if (topInput->textureDesc.pixelFormat != OP_PixelFormat::BGRA8Fixed)
	{
		myError = "CUDA Kernel is currently only written to handle 8-bit BGRA input textures.";
		return;
	}
	if (topInput->textureDesc.width != InputWidth || topInput->textureDesc.height != InputHeight)
	{
		myError = "Input texture must be 720x720";
		return;
	}

	// This example only takes RGB data
	const int NumComponents = 3;
	const int BytesPerComponent = sizeof(float);
	const int TensorSize = InputWidth * InputHeight * NumComponents * BytesPerComponent;
	if (!mySourceData || !mySourceTensor)
	{
		const std::array<int64_t, 4> shape = { 1, 3, InputHeight, InputWidth };
		if (!allocCUDAAndValue(&mySourceData, &mySourceTensor, TensorSize, shape))
			return;
	}
	if (!myDestData || !myDestTensor)
	{
		const std::array<int64_t, 4> shape = { 1, 3, InputHeight, InputWidth };
		if (!allocCUDAAndValue(&myDestData, &myDestTensor, TensorSize, shape))
			return;
	}

	OP_CUDAAcquireInfo acquireInfo;
	acquireInfo.stream = myStream;
	const OP_CUDAArrayInfo* inputArray = topInput->getCUDAArray(acquireInfo, nullptr);

	TOP_CUDAOutputInfo info;
	info.textureDesc = topInput->textureDesc;
	info.stream = myStream;

	// Primary output
	const OP_CUDAArrayInfo* outputInfo = output->createCUDAArray(info, nullptr);
	if (!outputInfo)
	{
		myError =  "Failed to create CUDA array for output";
		return;
	}

	// Now that we have gotten all of the pointers to the OP_CUDAArrayInfos that we may want, we can tell the context
	// that we are going to start doing CUDA operations. This will cause the cudaArray members of the OP_CUDAArrayInfo
	// to get filled in with valid addresses.
	if (!myContext->beginCUDAOperations(nullptr))
	{
		myError = "Failed to beingCUDAOperations().";
		return;
	}

	if (!setupCudaSurface(&myInputSurface, inputArray->cudaArray))
	{
		myError = "Failed to setup input CUDA surface.";
		myContext->endCUDAOperations(nullptr);
		return;
	}
	if (!setupCudaSurface(&myOutputSurface, outputInfo->cudaArray))
	{
		myError = "Failed to setup output CUDA surface.";
		myContext->endCUDAOperations(nullptr);
		return;
	}

	auto cudaRes = doCopyRGBATToRGBPlanar(info.textureDesc.width, info.textureDesc.height,
											info.textureDesc.depth, info.textureDesc.texDim, myInputSurface, (float*)mySourceData, myStream);

	OrtStatus* status = nullptr;

	const char* inputNames[] = { "inputImage" };
	const char* outputNames[] = { "outputImage" };
	status = myORT->Run(mySession, myRunOptions, inputNames, (const OrtValue* const*)&mySourceTensor, 1, outputNames, 1, &myDestTensor);
	std::string extraErr;
	if (!checkORTStatus(status, &extraErr))
	{
		myContext->endCUDAOperations(nullptr);
		myError = "Run() for the model failed. ";
		myError += extraErr;
		return;
	}

	// Since the results of this model is a texture, we can copy the results directly back into the output texture.
	cudaRes = doCopyRGBPlanarToRGBA(info.textureDesc.width, info.textureDesc.height, info.textureDesc.depth, info.textureDesc.texDim,
									(float*)myDestData, myOutputSurface, myStream);

	// If the results from the model are something that should instead be accessed on the CPU,
	// such as face or skeleton tracking points, you will need to issue a download from 'myDestData'.
	// to CPU memory via cudaMemcpyAsync(). Use 'myStream' as the stream argument.
	// You'll either need to call cudaStreamSynchronize(myStream) to ensure the data is availalbe.
	// Or to avoid stalls, you could insert CUDA events after issuing the download, and only
	// consume the data once that event has occured.

	myContext->endCUDAOperations(nullptr);
}

int32_t
ONNXCandyStyleTOP::getNumInfoCHOPChans(void* reserved)
{
	// We return the number of channel we want to output to any Info CHOP
	// connected to the TOP. In this example we are just going to send one channel.
	return 0;
}

void
ONNXCandyStyleTOP::getInfoCHOPChan(int32_t index,
						OP_InfoCHOPChan* chan,
						void* reserved)
{
}

bool		
ONNXCandyStyleTOP::getInfoDATSize(OP_InfoDATSize* infoSize, void* reserved)
{
	infoSize->rows = 0;
	infoSize->cols = 0;
	return true;
}

void
ONNXCandyStyleTOP::getInfoDATEntries(int32_t index,
										int32_t nEntries,
										OP_InfoDATEntries* entries,
										void* reserved)
{
}

void
ONNXCandyStyleTOP::getErrorString(OP_String *error, void* reserved)
{
	error->setString(myError.c_str());
}

void
ONNXCandyStyleTOP::setupParameters(OP_ParameterManager* manager, void* reserved)
{
}

void
ONNXCandyStyleTOP::pulsePressed(const char* name, void* reserved)
{
}
