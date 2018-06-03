#include <math.h>

#include "TRPO.h"


// Utility function calculating the number of trainable parameters
size_t NumParamsCalc (size_t * LayerSize, size_t NumLayers) {
    
    size_t NumParams = 0;
    for (size_t i=0; i<NumLayers-1; ++i) {
        // Weight and Bias
        NumParams += LayerSize[i] * LayerSize[i+1] + LayerSize[i+1];
    }
    // Std
    NumParams += LayerSize[NumLayers-1];
    return NumParams;
}

// Utility Function Calculating the Max
static inline double max(double record, double cur) {
	double result = (record<fabs(cur)) ? fabs(cur) : record;
	return result;
}
