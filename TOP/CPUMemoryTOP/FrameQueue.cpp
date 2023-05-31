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

#include "FrameQueue.h"
#include <assert.h>

using namespace TD;

FrameQueue::FrameQueue(TOP_Context* context) :
	myContext(context)
{

}

FrameQueue::~FrameQueue()
{
	while (!myUpdatedBuffers.empty())
	{
		myUpdatedBuffers.pop_front();
	}
}

const int MaxQueueSize = 2;

OP_SmartRef<TOP_Buffer>
FrameQueue::getBufferToUpdate(uint64_t byteSize, TOP_BufferFlags flags)
{
	myLock.lock();
	
	OP_SmartRef<TOP_Buffer> buf;

	// If we've already reached the max queue size, replace the oldest buffer
	// instead of requesting a new one and growing the queue even more.
	if (myUpdatedBuffers.size() >= MaxQueueSize)
	{
		buf = myUpdatedBuffers.front().buf;
		myUpdatedBuffers.pop_front();

		// If the size of this buffer is way off or if the flags are wrong,
		// don't use it.
		if (buf->size < byteSize || buf->size > byteSize * 2 || buf->flags != flags)
		{
			buf.release();
		}
	}

	// If we don't have a buffer yet, allocate one
	if (!buf)
		buf = myContext->createOutputBuffer(byteSize, flags, nullptr);

	myLock.unlock();
	return buf;
}

void
FrameQueue::updateComplete(const BufferInfo& bufInfo)
{
	assert(bufInfo.buf);
	myLock.lock();
	myUpdatedBuffers.push_back(bufInfo);
	myLock.unlock();
}

void
FrameQueue::updateCancelled(OP_SmartRef<TOP_Buffer>* buf)
{
	myContext->returnBuffer(buf);
}

BufferInfo
FrameQueue::getBufferToUpload()
{
	myLock.lock();

	BufferInfo buf;
	if (!myUpdatedBuffers.empty())
	{
		buf = myUpdatedBuffers.front();
		myUpdatedBuffers.pop_front();
	}
	myLock.unlock();
	return buf;
}
