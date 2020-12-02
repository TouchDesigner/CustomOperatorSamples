#include "OneEuroImpl.h"
#include <cmath>

static constexpr double PI = 3.141592653589793238463;

class LowPassFilter
{
public:
    LowPassFilter() :
        myFirstFiltering{ true }, myHatXPrev{}, myHatX{}
    {

    }

    virtual ~LowPassFilter()
    {

    }

    double  
    filter(double value, double alpha)
    {
        if (myFirstFiltering)
        {
            myFirstFiltering = false;
            myHatXPrev = value;
        }
        myHatX = alpha * value + (1.0 - alpha) * myHatXPrev;
        myHatXPrev = myHatX;
        return myHatX;
    }

    double 
    getHatXPrev() 
    { 
        return myHatXPrev; 
    }

private:
    bool    myFirstFiltering;

    double  myHatXPrev;
    double  myHatX;
};

OneEuroImpl::OneEuroImpl(
        double rate, double minCutOff, double beta, double dCutOff
    ) :
    myFirstFiltering{true}, myRate{rate}, myMinCutOff{minCutOff}, 
    myBeta{beta}, myXFilt{new LowPassFilter()}, myDCutOff{dCutOff},
    myDxFilt{new LowPassFilter()}
{
    
} 

OneEuroImpl::~OneEuroImpl()
{
    delete myXFilt;
    delete myDxFilt;
}

void
OneEuroImpl::changeInput(double rate, double minCutOff, double beta, double dCutOff)
{
    myRate = rate;
    myMinCutOff = minCutOff;
    myBeta = beta;
    myDCutOff = dCutOff;
}

double 
OneEuroImpl::filter(double x)
{
    double  dx = myFirstFiltering ? 0.0 : (x - myXFilt->getHatXPrev()) * myRate;
    myFirstFiltering = false;
    double  edx = myDxFilt->filter(dx, alpha(myDCutOff));
    double cutOff = myMinCutOff + myBeta * std::abs(edx);
    return myXFilt->filter(x, alpha(cutOff));
}

double
OneEuroImpl::alpha(double cutoff)
{
    double  tau = 1.0 / (2 * PI * cutoff);
    double  te = 1.0 / myRate;
    return 1.0 / (1.0 + tau/te);
}
