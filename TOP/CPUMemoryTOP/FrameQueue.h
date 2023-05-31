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

#pragma once

#include <deque>
#include <queue>
#include <mutex>

#include "TOP_CPlusPlusBase.h"

class BufferInfo
{
public:
	TD::OP_SmartRef<TD::TOP_Buffer>		buf;
	TD::TOP_UploadInfo					uploadInfo;

};
class FrameQueue
{
public:
	FrameQueue(TD::TOP_Context* context);
	~FrameQueue();

	// Call this to get a buffer to fill with new buffer data.
	// You should call either updateComplete() or updateCancelled() when done with the buffer.
	// You can also call release() on the buffer to say you are done with it, but that won't
	// allow it to be re-used for another operation later on (possibly avoiding an allocation).
	// This may return nullptr if there is no buffer available for update.
	TD::OP_SmartRef<TD::TOP_Buffer>		getBufferToUpdate(uint64_t byteSize, TD::TOP_BufferFlags flags);

	// Takes ownership of the TOP_Buffer contained in BufferInfo, don't release it externally.
	void				updateComplete(const BufferInfo &bufInfo);

	// Call this to tell the class that the data from the last getBufferForUpdate()
	// did not get filled so it should not be queued for upload to the TOP
	void				updateCancelled(TD::OP_SmartRef<TD::TOP_Buffer> *buf);

	// If there is a new buffer to upload, BufferInfo.buf will not be nullptr.
	// You are the owner of BufferInfo.buf if this returns a non-nullptr
	BufferInfo			getBufferToUpload();

private:
	std::mutex				myLock;
	std::deque<BufferInfo>	myUpdatedBuffers;

	TD::TOP_Context*		myContext;
};
