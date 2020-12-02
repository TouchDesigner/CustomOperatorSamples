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
#ifndef __OneEuroImpl__
#define __OneEuroImpl__

class LowPassFilter;

/*
Implementation of the 1€ Filter described in the paper:
    - Casiez, G., Roussel, N. and Vogel, D. (2012). 1€ Filter: A Simple 
Speed-based Low-pass Filter for Noisy Input in Interactive Systems. 
Proceedings of the ACM Conference on Human Factors in Computing Systems 
(CHI '12). Austin, Texas (May 5-12, 2012). New York: ACM Press, pp. 2527-2530.
*/

class OneEuroImpl
{
public:
    OneEuroImpl(double rate, double minCutOff, double beta, double dCutOff);
    virtual ~OneEuroImpl();

    void            changeInput(double rate, double minCutOff, double beta, double dCutOff);
    
    double          filter(double value);

    double          alpha(double cutOff);

private:
    bool            myFirstFiltering;
    double          myRate;
    double          myMinCutOff;
    double          myBeta;
    LowPassFilter*  myXFilt;
    double          myDCutOff;
    LowPassFilter*  myDxFilt;
};
#endif
