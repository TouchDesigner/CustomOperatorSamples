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

#include "CPUMemoryTOP.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <cmath>
#include <random>
#include <chrono>

// Uncomment this if you want to run an example that fills the data using threading
//#define THREADING_EXAMPLE

// The threading example can run in two modes. One where the producer is continually
// producing new frames that the consumer (main thread) picks up when able.
// This more is useful for things such as external device input. This is the default
// mode.

// The second mode can be instead used by defining THREADING_SINGLED_PRODUCER.
// (I.e Uncommenting the below line)
// In this mode the main thread will signal to the producer thread to generate a new
// frame each time it consumes a frame.
// Assuming the producer will generate a new frame in time before execute() gets called
// again this gives a better 1:1 sync between producing and consuming frames.

//#define THREADING_SIGNALED_PRODUCER


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
	info->executeMode = TOP_ExecuteMode::CPUMemWriteOnly;

	// The opType is the unique name for this TOP. It must start with a 
	// capital A-Z character, and all the following characters must lower case
	// or numbers (a-z, 0-9)
	info->customOPInfo.opType->setString("Cpumemsample");

	// The opLabel is the text that will show up in the OP Create Dialog
	info->customOPInfo.opLabel->setString("CPU Mem Sample");

	// Will be turned into a 3 letter icon on the nodes
	info->customOPInfo.opIcon->setString("CPM");

	// Information about the author of this OP
	info->customOPInfo.authorName->setString("Gabriel Robels");
	info->customOPInfo.authorEmail->setString("support@derivative.ca");

	// This TOP works with 0 or 1 inputs connected
	info->customOPInfo.minInputs = 0;
	info->customOPInfo.maxInputs = 1;
}

DLLEXPORT
TOP_CPlusPlusBase*
CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per TOP that is using the .dll
	return new CPUMemoryTOP(info);
}

DLLEXPORT
void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context *context)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the TOP using that instance is deleted, or
	// if the TOP loads a different DLL
	delete (CPUMemoryTOP*)instance;
}

};



CPUMemoryTOP::CPUMemoryTOP(const OP_NodeInfo* info) : 
	myNodeInfo(info),
	myThread(nullptr),
	myThreadShouldExit(false),
	myStartWork(false)
{
	myExecuteCount = 0;
	myStep = 0.0;
	myBrightness = 1.0;
}

CPUMemoryTOP::~CPUMemoryTOP()
{
#ifdef THREADING_EXAMPLE
	if (myThread)
	{
		myThreadShouldExit.store(true);
		// Incase the thread is sleeping waiting for a signal
		// to create more work, wake it up
		startMoreWork();
		if (myThread->joinable())
		{
			myThread->join();
		}
		delete myThread;
	}
#endif

}

void
CPUMemoryTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs* inputs, void* reserved1)
{
	ginfo->cookEveryFrame = true;
    ginfo->memPixelType = OP_CPUMemPixelType::RGBA32Float;
}

bool
CPUMemoryTOP::getOutputFormat(TOP_OutputFormat* format, const OP_Inputs* inputs, void* reserved1)
{
	// In this function we could assign variable values to 'format' to specify
	// the pixel format/resolution etc that we want to output to.
	// If we did that, we'd want to return true to tell the TOP to use the settings we've
	// specified.
	// In this example we'll return false and use the TOP's settings
	return false;
}


void
CPUMemoryTOP::execute(TOP_OutputFormatSpecs* output,
						const OP_Inputs* inputs,
						TOP_Context *context,
						void* reserved1)
{
	myExecuteCount++;


#ifdef THREADING_EXAMPLE
	mySettingsLock.lock();
#endif

	double speed = inputs->getParDouble("Speed");
	mySpeed = speed;

	myBrightness = inputs->getParDouble("Brightness");

	// See comments at the top of this file to information about the threading
	// example mode for this project.
#ifdef THREADING_EXAMPLE
	mySettingsLock.unlock();

	// This syncs up the buffers in the frame queue with what the
	// node is offering
	myFrameQueue.sync(output);

	if (!myThread)
	{
		myThread = new std::thread(
			[this]()
			{
				std::random_device rd;
				std::mt19937 mt(rd());

				// We are going to generate new frame data at irregular interval
				std::uniform_real_distribution<double> dist(10.0, 40.0);

				// Exit when our owner tells us to
				while (!this->myThreadShouldExit)
				{
#ifdef THREADING_SIGNALED_PRODUCER
					this->waitForMoreWork();
					// We may be waking up because the owner is trying to shut down
					if (myThreadShouldExit)
					{
						break;
					}
#else
					auto begin = std::chrono::steady_clock::now();
#endif

					int width, height;
					void *buf = this->myFrameQueue.getBufferForUpdate(&width, &height);

					// If there is a buffer to update
					if (buf)
					{
						this->mySettingsLock.lock();
						myStep += mySpeed;
						double brightness = myBrightness;
						this->mySettingsLock.unlock();

						CPUMemoryTOP::fillBuffer((float*)buf, width, height, myStep, brightness);


						this->myFrameQueue.updateComplete();
					}

#ifndef THREADING_SIGNALED_PRODUCER
					auto end = std::chrono::steady_clock::now();
					auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

					// Sleep for a 15.5ms. We want our loops to be 16.666ms, so this give some wiggle wrong.
					// In almost all cases we wouldn't be manually sleeping, but instead sleeping
					// using a function from the SDK of the device that we are reading.
					std::this_thread::sleep_for(std::chrono::microseconds(15500 - duration));
#endif
				}
			});
	}

	// Tries to assign a buffer to be uploaded to the TOP
	myFrameQueue.sendBufferForUpload(output);

#ifdef THREADING_SIGNALED_PRODUCER
	// Tell the thread to make another frame
	startMoreWork();
#endif

#else

	myStep += speed;
    int textureMemoryLocation = 0;
    float* mem = (float*)output->cpuPixelData[textureMemoryLocation];

	fillBuffer(mem, output->width, output->height, myStep, myBrightness);

	// Tell the TOP which buffer to upload. In this simple example we are always filling and uploading buffer 0
    output->newCPUPixelDataLocation = textureMemoryLocation;

#endif

}

void
CPUMemoryTOP::fillBuffer(float *mem, int width, int height, double step, double brightness)
{

	int xstep = (int)(fmod(step, width));
	int ystep = (int)(fmod(step, height));

	if (xstep < 0)
		xstep += width;

	if (ystep < 0)
		ystep += height;




    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float* pixel = &mem[4*(y*width + x)];

			// RGBA
            pixel[0] = (x > xstep) * (float)brightness;
            pixel[1] = (y > ystep) * (float)brightness;
			pixel[2] = ((float)(xstep % 50) / 50.0f) * (float)brightness;
            pixel[3] = 1;
        }
    }
}

void
CPUMemoryTOP::startMoreWork()
{
	{
		std::unique_lock<std::mutex> lck(myConditionLock);
		myStartWork = true;
	}
	myCondition.notify_one();
}

void
CPUMemoryTOP::waitForMoreWork()
{
	std::unique_lock<std::mutex> lck(myConditionLock);
	myCondition.wait(lck, [this]() { return this->myStartWork.load(); });
	myStartWork = false;
}

int32_t
CPUMemoryTOP::getNumInfoCHOPChans(void *reserved1)
{
	// We return the number of channel we want to output to any Info CHOP
	// connected to the TOP. In this example we are just going to send one channel.
	return 2;
}

void
CPUMemoryTOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void* reserved1)
{
	// This function will be called once for each channel we said we'd want to return
	// In this example it'll only be called once.

	if (index == 0)
	{
		chan->name->setString("executeCount");
		chan->value = (float)myExecuteCount;
	}

	if (index == 1)
	{
		chan->name->setString("step");
		chan->value = (float)myStep;
	}
}

bool		
CPUMemoryTOP::getInfoDATSize(OP_InfoDATSize* infoSize, void* reserved1)
{
	infoSize->rows = 2;
	infoSize->cols = 2;
	// Setting this to false means we'll be assigning values to the table
	// one row at a time. True means we'll do it one column at a time.
	infoSize->byColumn = false;
	return true;
}

void
CPUMemoryTOP::getInfoDATEntries(int32_t index,
								int32_t nEntries,
								OP_InfoDATEntries* entries,
								void *reserved1)
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

	if (index == 1)
	{
#ifdef _WIN32
		strcpy_s(tempBuffer, "step");
#else // macOS
        strlcpy(tempBuffer, "step", sizeof(tempBuffer));
#endif
		entries->values[0]->setString(tempBuffer);

#ifdef _WIN32
		sprintf_s(tempBuffer, "%g", myStep);
#else // macOS
        snprintf(tempBuffer, sizeof(tempBuffer), "%g", myStep);
#endif
		entries->values[1]->setString(tempBuffer);
	}
}

void
CPUMemoryTOP::setupParameters(OP_ParameterManager* manager, void *reserved1)
{
	// brightness
	{
		OP_NumericParameter	np;

		np.name = "Brightness";
		np.label = "Brightness";
		np.defaultValues[0] = 1.0;

		np.minSliders[0] =  0.0;
		np.maxSliders[0] =  1.0;

		np.minValues[0] = 0.0;
		np.maxValues[0] = 1.0;

		np.clampMins[0] = true;
		np.clampMaxes[0] = true;
		
		OP_ParAppendResult res = manager->appendFloat(np);
		assert(res == OP_ParAppendResult::Success);
	}

	// speed
	{
		OP_NumericParameter	np;

		np.name = "Speed";
		np.label = "Speed";
		np.defaultValues[0] = 1.0;
		np.minSliders[0] = -10.0;
		np.maxSliders[0] =  10.0;
		
		OP_ParAppendResult res = manager->appendFloat(np);
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
CPUMemoryTOP::pulsePressed(const char* name, void *reserved1)
{
	if (!strcmp(name, "Reset"))
	{
		myStep = 0.0;
	}


}

