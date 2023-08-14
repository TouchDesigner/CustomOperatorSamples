#include "ThreadManager.h"

#include "FilterWork.h"


#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>


ThreadManager::ThreadManager() :
	myStatus{ ThreadStatus::Waiting }, myOutBuffer{nullptr}, myDownRes{nullptr},
	myThread{}, myBufferMutex{}, myBufferCV{},
	myThreadShouldExit{false}, myInWidth{}, myInHeight{},
	myOutWidth{}, myOutHeight{}, myDoDither{ false }, myBitsPerColor{ 8 }, myContext{ nullptr }, myUploadInfo{}
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

	std::unique_lock<std::mutex> bufferlock(myBufferMutex);
	if(myContext && myOutBuffer)
		myContext->returnBuffer(&myOutBuffer);
	bufferlock.unlock();

	delete myThread;
}

void 
ThreadManager::sync(bool doDither, int bitsPerColor, int inWidth, int inHeight, 
																			const OP_SmartRef<OP_TOPDownloadResult> downRes, TD::TOP_Context* context)
{
	std::unique_lock<std::mutex> bufferlock(myBufferMutex);
	myDoDither = doDither;
	myBitsPerColor = bitsPerColor;
	myInWidth = inWidth;
	myInHeight = inHeight;

	myContext = context;
	myDownRes = downRes;
	myUploadInfo.textureDesc = myDownRes->textureDesc;
	myOutWidth = myUploadInfo.textureDesc.width;
	myOutHeight = myUploadInfo.textureDesc.height;
	myStatus.store(ThreadStatus::Ready);
	bufferlock.unlock();

	myBufferCV.notify_all();

}


void 
ThreadManager::threadFn()
{
	while (!myThreadShouldExit)
	{
		std::unique_lock<std::mutex>		bufferLock(myBufferMutex);
		myBufferCV.wait(bufferLock, [this] { return myStatus.load() == ThreadStatus::Ready || myThreadShouldExit; });
		if (myThreadShouldExit)
		{
			if (myDownRes)
				myDownRes.release();
			return;
		}

		myStatus.store(ThreadStatus::Busy);

		const int							outwidth = myOutWidth;
		const int							outheight = myOutHeight;
		const int							inwidth = myInWidth;
		const int							inheight = myInHeight;
		const bool							doDither = myDoDither;
		const int							bitsPerColor = myBitsPerColor;
		TOP_Context*					context = myContext;
		OP_SmartRef<OP_TOPDownloadResult> downRes = myDownRes;

		size_t byteSize = inwidth * inheight * sizeof(uint32_t);
		myOutBuffer = context->createOutputBuffer(byteSize, TOP_BufferFlags::None, nullptr);

		uint32_t* outbuf = (uint32_t*)myOutBuffer->data;


		uint32_t* inBuffer = (uint32_t*)downRes->getData();

		Filter::doFilterWork(inBuffer, inwidth, inheight, outbuf, outwidth, outheight, doDither, bitsPerColor);
		myDownRes.release();
		myStatus.store(ThreadStatus::Done);
		bufferLock.unlock();

		myBufferCV.notify_one();
	}
}


void
ThreadManager::popOutBuffer(OP_SmartRef<TOP_Buffer>& outBuffer, TD::TOP_UploadInfo& info)
{
	std::lock_guard<std::mutex> bufferLock(myBufferMutex);
	outBuffer = std::move(myOutBuffer);
	info = std::move(myUploadInfo);
	myStatus.store(ThreadStatus::Waiting);
}

ThreadStatus
ThreadManager::getStatus()
{
	return myStatus.load();
}