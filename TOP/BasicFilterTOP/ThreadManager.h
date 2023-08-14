#ifndef __ThreadManager__
#define __ThreadManager__

#include "TOP_CPlusPlusBase.h"

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>

using namespace TD;

// class Parameters;
class OP_Inputs;


enum class ThreadStatus
{
	Busy,
	Ready,
	Done,
	Waiting
};

class ThreadManager
{
public:
	ThreadManager();

	~ThreadManager();

	void	sync(bool doDither, int bitsPerColor, int inWidth, int inHeight, const OP_SmartRef<OP_TOPDownloadResult> downRes, TD::TOP_Context* context);


	void	popOutBuffer(OP_SmartRef<TOP_Buffer>& outBuffer, TD::TOP_UploadInfo& info);
	
	ThreadStatus getStatus();
private:
	void	threadFn();

	std::atomic<ThreadStatus>			myStatus;

	// Out buffer resource
	OP_SmartRef<OP_TOPDownloadResult>	myDownRes;
	OP_SmartRef<TOP_Buffer>				myOutBuffer;

	// Thread and Sync variables
	std::thread*			myThread;
	std::mutex				myBufferMutex;
	std::condition_variable myBufferCV;
	std::atomic_bool		myThreadShouldExit;

	// Parameters saved
	int						myInWidth;
	int						myInHeight;
	int						myOutWidth;
	int						myOutHeight;
	bool					myDoDither;
	int						myBitsPerColor;
	TD::TOP_Context*		myContext;
	TD::TOP_UploadInfo		myUploadInfo;
};

#endif
