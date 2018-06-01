#ifndef TRPO_H
#define TRPO_H


typedef struct {

    //////////////////// For CPU and FPGA ////////////////////

    // Model Parameter File Name - weight, bias, std.
    char * ModelFile;
    
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

#endif
