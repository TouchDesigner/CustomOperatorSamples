#include "ThreadManager.h"

#include "FilterWork.h"


#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>



ThreadManager::ThreadManager() :
	myStatus{ ThreadStatus::Done }, myOutBuffer{}, myInBuffer{},
	myThread{}, myBufferMutex{}, myParmsMutex{}, myBufferCV{}, 
	myThreadShouldExit{false}, myInWidth{}, myInHeight{},
	myOutWidth{}, myOutHeight{}, myDoDither{false}, myBitsPerColor{8}
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
}

void 
ThreadManager::syncParms(bool doDither, int bitsPerColor, int inWidth, int inHeight, int outWidth, int outHeight, const TD::OP_Inputs* inputs)
{
	const std::lock_guard<std::mutex> lock(myParmsMutex);
	myDoDither = doDither;
	myBitsPerColor = bitsPerColor;
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
	lock.unlock();
	myStatus = ThreadStatus::Ready;
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
		const int							outwidth = myOutWidth;
		const int							outheight = myOutHeight;
		const int							inwidth = myInWidth;
		const int							inheight = myInHeight;
		const bool							doDither = myDoDither;
		const int							bitsPerColor = myBitsPerColor;
		parmsLock.unlock();

		Filter::doFilterWork(myInBuffer, inwidth, inheight, myOutBuffer, outwidth, outheight, doDither, bitsPerColor);
		delete[] myInBuffer;
		myInBuffer = nullptr;
		myStatus = ThreadStatus::Done;
		bufferLock.unlock();
		myBufferCV.notify_one();
	}
}
