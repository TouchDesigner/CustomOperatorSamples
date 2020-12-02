#include "ThreadManager.h"

#include "FilterWork.h"
#include "Parameters.h"

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>

ThreadManager::ThreadManager() :
	myStatus{ ThreadStatus::Done }, myOutBuffer{}, myInBuffer{},
	myThread{}, myBufferMutex{}, myParmsMutex{}, myBufferCV{}, 
	myThreadShouldExit{false}, myInWidth{}, myInHeight{},
	myOutWidth{}, myOutHeight{}, myParms{new Parameters()}
{
	myThread = new std::thread([this] { threadFn(); });
}


ThreadManager::~ThreadManager()
{
	myThreadShouldExit.store(true);
	myBufferCV.notify_all();
	if (myThread->joinable())
	{
		myThread->join();
	}
	delete myThread;
	delete myParms;
}

void 
ThreadManager::syncParms(const Parameters& parms, int inWidth, int inHeight, int outWidth, int outHeight)
{
	const std::lock_guard<std::mutex> lock(myParmsMutex);
	*myParms = parms;
	myInWidth = inWidth;
	myInHeight = inHeight;
	myOutWidth = outWidth;
	myOutHeight = outHeight;
}

void 
ThreadManager::syncBuffer(uint32_t* inBuffer, uint32_t* outBuffer)
{
	std::unique_lock<std::mutex>	lock(myBufferMutex);
	myBufferCV.wait(lock, [this] { return myStatus == ThreadStatus::Done; });
	myOutBuffer = outBuffer;
	myInBuffer = new uint32_t[myInWidth * myInHeight];
	memcpy(myInBuffer, inBuffer, myInWidth * myInHeight * sizeof(uint32_t));
	myStatus = ThreadStatus::Ready;
	lock.unlock();
	myBufferCV.notify_all();
}

void 
ThreadManager::threadFn()
{
	while (!myThreadShouldExit)
	{
		std::unique_lock<std::mutex>		bufferLock(myBufferMutex);
		myBufferCV.wait(bufferLock, [this] { return myStatus == ThreadStatus::Ready || myThreadShouldExit; });
		if (myThreadShouldExit)
		{
			if (myInBuffer)
				delete[] myInBuffer;
			myInBuffer = nullptr;
			return;
		}

		myStatus = ThreadStatus::Busy;

		std::unique_lock<std::mutex>		parmsLock(myParmsMutex);
		Parameters							parms = *myParms;
		const int							outwidth = myOutWidth;
		const int							outheight = myOutHeight;
		const int							inwidth = myInWidth;
		const int							inheight = myInHeight;
		parmsLock.unlock();

		Filter::doFilterWork(myInBuffer, inwidth, inheight, myOutBuffer, outwidth, outheight, parms);
		delete[] myInBuffer;
		myInBuffer = nullptr;
		myStatus = ThreadStatus::Done;
		bufferLock.unlock();
		myBufferCV.notify_one();
	}
}
