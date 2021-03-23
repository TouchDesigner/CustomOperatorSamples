/* Shared Use License: This file is owned by Derivative Inc. (Derivative)
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

#ifndef __ContoursTOP__
#define __ContoursTOP__

#include "TOP_CPlusPlusBase.h"

#include <opencv2\core.hpp>
#include <string>

struct Parameters;

/*
This example implements a TOP to find contours and do image segmentation using openCV's cuda functionallity.

It takes the following parameters:
	- Mode:    One of [External, List, CComp, Tree, Floodfill].
	- Method:   One of [None, Simple, Teh-Chin L1, Teh-Chin KCos].
	- Apply Watershed:  If On, run watershed algorithm to segment the image.
	- Select Object:    If On, a single object will be outputted.
	- Object:   Which object to output, -1 outputs the boundary between sections.

This TOP takes one input which must be 8 bit single channel.

This TOP outputs an Info CHOP with the following channels:
	- Number of objects: The number of contours or segments found in the image.
*/

// To get more help about these functions, look at TOP_CPlusPlusBase.h
class ContoursTOP : public TOP_CPlusPlusBase
{
public:
    ContoursTOP(const OP_NodeInfo *info);
    virtual ~ContoursTOP();

    virtual void		getGeneralInfo(TOP_GeneralInfo*, const OP_Inputs*, void* reserved) override;

    virtual bool		getOutputFormat(TOP_OutputFormat*, const OP_Inputs*, void* reserved) override;

    virtual void		execute(TOP_OutputFormatSpecs*, const OP_Inputs*, TOP_Context*, void* reserved) override;

	virtual void		setupParameters(OP_ParameterManager*, void* reserved) override;

	virtual int32_t		getNumInfoCHOPChans(void* reserved) override;

	virtual void		getInfoCHOPChan(int32_t index, OP_InfoCHOPChan*, void* reserved) override;

private:
	void                inputTopToMat(const OP_Inputs*);

	void                secondInputTopToMat(const OP_Inputs*);

	void 		    cvMatToOutput(TOP_OutputFormatSpecs*) const;

	Parameters*		myParms;

	cv::Mat*		myFrame;
	cv::Mat*		mySecondInputFrame;
	int				myNComponents;
};

#endif
