#ifndef TRPO_H
#define TRPO_H

#include <stddef.h>

typedef struct {

    //////////////////// For CPU and FPGA ////////////////////

    // Model Parameter File Name - weight, bias, std.
    char * ModelFile;

    // Baseline File Name - weight, bias, std.
    char * BaselineFile;    
    
    // Simulation Data File Name - probability and observation
    char * DataFile;
    
    // Number of Layers in the network: [Input] --> [Hidden 1] --> [Hidden 2] --> [Output] is 4 layers.
    size_t NumLayers;
    
    // Activation Function of each layer: 't' = tanh, 'l' = linear (y=x), 's' = sigmoid
    // Activation Function used in the Final Layer in modular_rl: 'o' y = 0.1x
    char * AcFunc;
    
    // Number of nodes in each layer: from [Input] to [Output]
    // InvertedPendulum-v1: 4, 64, 64, 1
    // Humanoid-v1: 376, 64, 64, 17    
    size_t * LayerSize;
    
    // Number of Samples
    size_t NumSamples;
    
    // Conjugate Gradient Damping Factor 
    double CG_Damping;
    
    //////////////////// For FPGA Only ////////////////////
    
    // LayerSize used in FPGA, with stream padding
    size_t * PaddedLayerSize;
    
    // Number of Blocks for Each Layer, i.e. Parallelism Factor
    size_t * NumBlocks;
    
    
} TRPOparam;

typedef struct {

    // Paramaters
    size_t NumLayers;
    size_t ObservSpaceDim;
    size_t NumEpBatch;
    size_t EpLen;
    size_t NumSamples;
    size_t NumParams;
    int PaddedParams;
    char * AcFunc;
    
    // For Forward Propagation and Back Propagation
    size_t * LayerSizeBase;
    double ** WBase;
    double ** BBase;
    double ** LayerBase;
    double ** GWBase;
    double ** GBBase;
    double ** GLayerBase;
    
    // Training Data
    double * Observ;
    double * Target;    // The prediction target
    double * Predict;   // The prediction

} TRPOBaselineParam;


// Utility function calculating the number of trainable parameters
size_t NumParamsCalc (size_t * LayerSize, size_t NumLayers);

// Utility Function Calculating the Max
static inline double max(double record, double cur);

// Original FVP Computation Function on CPU
// FP + BP + Pearlmutter FP + Pearlmutter BP
double FVP(TRPOparam param, double *Result, double *Input);

// Fast FVP Computation Function on CPU
// Combined FP + Pearlmutter BP
double FVPFast(TRPOparam param, double *Result, double *Input, size_t NumThreads);

// CG Computation on CPU
double CG(TRPOparam param, double *Result, double *b, size_t MaxIter, double ResidualTh, size_t NumThreads);

// FVP Computation on FPGA
double FVP_FPGA(TRPOparam param, double *Result, double *Input);

// CG Computation on FPGA
double CG_FPGA(TRPOparam param, double *Result, double *b, size_t MaxIter, double ResidualTh, size_t NumThreads);

// TRPO Update on CPU
double TRPO_Update(TRPOparam param, double *Result, size_t NumThreads);

// TRPO All-In-One
double RunTraining (TRPOparam param, const int NumIter, const size_t NumThreads);

#endif
