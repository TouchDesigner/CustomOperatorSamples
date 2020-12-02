#ifndef __ThreadManager__
#define __ThreadManager__

#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>

struct Parameters;

enum class ThreadStatus
{
	Busy,
	Ready,
	Done
};

class ThreadManager
{
public:
	ThreadManager();

	~ThreadManager();

	void	syncParms(const Parameters& parms, int inWidth, int inHeight, int outWidth, int outHeight);

	void	syncBuffer(uint32_t* inBuffer, uint32_t* outBuffer);

private:
	void	threadFn();

	ThreadStatus			myStatus;

	// Out buffer resource
	uint32_t*				myInBuffer;
	uint32_t*				myOutBuffer;

	// Thread and Sync variables
	std::thread*			myThread;
	std::mutex				myBufferMutex;
	std::mutex				myParmsMutex;
	std::condition_variable myBufferCV;
	std::atomic_bool		myThreadShouldExit;

	// Parameters saved
	int						myInWidth;
	int						myInHeight;
	int						myOutWidth;
	int						myOutHeight;
	Parameters*				myParms;
};

#endif
