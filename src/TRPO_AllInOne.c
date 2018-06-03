#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "omp.h"

#include "TRPO.h"


double TRPO(TRPOparam param, double *Result, double *Input, size_t NumThreads) {

    //////////////////// Remarks ////////////////////

    // This function computes the Fisher-Vector Product using Pearlmutter Algorithm
    // This version is customised to the case that KL is used as loss function
    // Input: the vector to be multiplied with the Fisher Information Matrix
    // Result: the Fisher-Vector Product
    // Remarks: The length of Input and Result must be the number of all trainable parameters in the network

    // Step1: Combined forward propagation
    // Step2: Pearlmutter backward propagation


    //////////////////// Read Parameters ////////////////////

    // OpenMP Settings
    omp_set_num_threads(NumThreads);

    // Assign Parameters
    const size_t NumLayers  = param.NumLayers;
    char * AcFunc           = param.AcFunc;
    size_t * LayerSize      = param.LayerSize;
    const size_t NumSamples = param.NumSamples;
    char * ModelFile        = param.ModelFile;
    char * DataFile         = param.DataFile;
    const double CG_Damping = param.CG_Damping;

    // Dimension of Observation Space
    const size_t ObservSpaceDim = LayerSize[0];
    // Dimension of Action Space
    const size_t ActionSpaceDim = LayerSize[NumLayers-1];

    // iterator when traversing through input vector and result vector
    size_t pos;


    //////////////////// Memory Allocation - Neural Network ////////////////////
    
    // W[i]: Weight Matrix from Layer[i] to Layer[i+1]
    // B[i]: Bias Vector from Layer[i] to Layer[i+1]
    // Item (j,k) in W[i] refers to the weight from Neuron #j in Layer[i] to Neuron #k in Layer[i+1]
    // Item B[k] is the bias of Neuron #k in Layer[i+1]
    double * W [NumLayers-1];
    double * B [NumLayers-1];
    for (size_t i=0; i<NumLayers-1; ++i) {
        W[i] = (double *) calloc(LayerSize[i]*LayerSize[i+1], sizeof(double));
        B[i] = (double *) calloc(LayerSize[i+1], sizeof(double));
    }


    //////////////////// Memory Allocation - Input Vector ////////////////////

    // The Input Vector is to be multiplied with the Hessian Matrix of KL to derive the Fisher Vector Product
    // There is one-to-one correspondence between the input vector and all trainable parameters in the neural network
    // As a result, the shape of the Input Vector is the same as that of the parameters in the model
    // The only difference is that the Input Vector is stored in a flattened manner
    // There is one-to-one correspondence between: VW[i] and W[i], VB[i] and B[i], VStd[i] and Std[i]
    double * VW [NumLayers-1];
    double * VB [NumLayers-1];
    for (size_t i=0; i<NumLayers-1; ++i) {
        VW[i] = (double *) calloc(LayerSize[i]*LayerSize[i+1], sizeof(double));
        VB[i] = (double *) calloc(LayerSize[i+1], sizeof(double));
    }
    
    // Allocate Memory for Input Vector corresponding to LogStd
    double * VLogStd = (double *) calloc(ActionSpaceDim, sizeof(double));


    //////////////////// Memory Allocation - Simulation Data ////////////////////

    // Allocate Memory for Observation and Probability Mean
    // Observ: list of observations - corresponds to ob_no in modular_rl
    // Mean: list of probablity mean values - corresponds to the 'mean' part of prob_np in modular_rl
    // Remarks: due to the specific setting of the experienments in the TRPO paper,
    //          Std is the same for all samples in each simulation iteration,
    //          so we just allocate Std memory space for one sample and use it for all samples.
    //          The general case should be another vector of Std with size NumSamples*ActionSpaceDim
    double * Observ    = (double *) calloc(NumSamples*ObservSpaceDim, sizeof(double));
    double * Mean      = (double *) calloc(NumSamples*ActionSpaceDim, sizeof(double));
    double * Std       = (double *) calloc(ActionSpaceDim, sizeof(double));
    double * Action    = (double *) calloc(NumSamples*ActionSpaceDim, sizeof(double));
    double * Advantage = (double *) calloc(NumSamples, sizeof(double));
    
    
    //////////////////// Memory Allocation - Ordinary Forward Propagation ////////////////////

    // Layer[i] : Memory of each layer's outputs, i.e. y_i
    double * Layer  [NumLayers];
    for (size_t i=0; i<NumLayers; ++i) {
        Layer[i]  = (double *) calloc(LayerSize[i], sizeof(double));
    }


    //////////////////// Memory Allocation - Pearlmutter Forward and Backward Propagation ////////////////////

    // RyLayer[i]: R{} of each layer's outputs, i.e. R{y_i}
    // RxLayer[i]: R{} of each layer's pre-activated outputs, i.e. R{x_i}
    // RGLayer[I]: R{} Gradient of KL w.r.t. the pre-activation values in Layer[i], i.e. R{d(KL)/d(x_i)}
    double * RyLayer [NumLayers];
    double * RxLayer [NumLayers];
    double * RGLayer [NumLayers];
    for (size_t i=0; i<NumLayers; ++i) {
        RyLayer[i] = (double *) calloc(LayerSize[i], sizeof(double));
        RxLayer[i] = (double *) calloc(LayerSize[i], sizeof(double));
        RGLayer[i] = (double *) calloc(LayerSize[i], sizeof(double));
    }

    // RGW[i]: R{} Gradient of KL w.r.t. to Neural Network Weight W[i], i.e. R{d(KL)/d(W[i])}
    // RGB[i]: R{} Gradient of KL w.r.t. to Neural Network Bias B[i], i.e. R{d(KL)/d(B[i])}
    // There is one-to-one correspondence between: RGW[i] and W[i], RGB[i] and B[i]
    double * RGW [NumLayers-1];
    double * RGB [NumLayers-1];
    for (size_t i=0; i<NumLayers-1; ++i) {
        RGW[i] = (double *) calloc(LayerSize[i]*LayerSize[i+1], sizeof(double));
        RGB[i] = (double *) calloc(LayerSize[i+1], sizeof(double));
    }
    
    
    //////////////////// Load Neural Network ////////////////////
    
    // Open Model File that contains Weights, Bias and std
    FILE *ModelFilePointer = fopen(ModelFile, "r");
    if (ModelFilePointer==NULL) {
        fprintf(stderr, "[ERROR] Cannot open Model File [%s]. \n", ModelFile);
        return -1;
    }
    
    // Read Weights and Bias from file
    for (size_t i=0; i<NumLayers-1; ++i) {
        // Reading Weights W[i]: from Layer[i] to Layer[i+1]
        size_t curLayerDim  = LayerSize[i];
        size_t nextLayerDim = LayerSize[i+1];
        for (size_t j=0; j<curLayerDim;++j) {
            for (size_t k=0; k<nextLayerDim; ++k) {
                fscanf(ModelFilePointer, "%lf", &W[i][j*nextLayerDim+k]);
            }
        }
        // Reading Bias B[i]: from Layer[i] to Layer[i+1]
        for (size_t k=0; k<nextLayerDim; ++k) {
            fscanf(ModelFilePointer, "%lf", &B[i][k]);
        }
    }

    // Read LogStd from file
    // Remarks: actually this LogStd will be overwritten by the Std from the datafile
    for (size_t k=0; k<ActionSpaceDim; ++k) {
        fscanf(ModelFilePointer, "%lf", &Std[k]);
    }

    // Close Model File
    fclose(ModelFilePointer);
    
    
    //////////////////// Load Input Vector and Init Result Vector ////////////////////
    
    pos = 0;
    for (size_t i=0; i<NumLayers-1; ++i) {
        size_t curLayerDim  = LayerSize[i];
        size_t nextLayerDim = LayerSize[i+1];
        for (size_t j=0; j<curLayerDim;++j) {
            for (size_t k=0; k<nextLayerDim; ++k) {
                VW[i][j*nextLayerDim+k] = Input[pos];
                Result[pos] = 0;
                pos++;
            }
        }
        for (size_t k=0; k<nextLayerDim; ++k) {
            VB[i][k] = Input[pos];
            Result[pos] = 0;
            pos++;
        }
    }
    for (size_t k=0; k<ActionSpaceDim; ++k) {
        VLogStd[k] = Input[pos];
        Result[pos] = 0;
        pos++;
    }
    
    
    //////////////////// Load Simulation Data ////////////////////
    
    // Open Data File that contains Mean, std and Observation
    FILE *DataFilePointer = fopen(DataFile, "r");
    if (DataFilePointer==NULL) {
        fprintf(stderr, "[ERROR] Cannot open Data File [%s]. \n", DataFile);
        return -1;
    }
    
    // Read Mean, Std and Observation
    // Remarks: Std is the same for all samples, and appears in every line in the data file
    //          so we are writing the same Std again and again to the same place.
    for (size_t i=0; i<NumSamples; ++i) {
        // Read Mean
        for (size_t j=0; j<ActionSpaceDim; ++j) {
            fscanf(DataFilePointer, "%lf", &Mean[i*ActionSpaceDim+j]);
        }
        // Read Std
        for (size_t j=0; j<ActionSpaceDim; ++j) {
            fscanf(DataFilePointer, "%lf", &Std[j]);
        }
        // Read Observation
        for (size_t j=0; j<ObservSpaceDim; ++j) {
            fscanf(DataFilePointer, "%lf", &Observ[i*ObservSpaceDim+j]);
        }
        // Read Action
        for (size_t j=0; j<ActionSpaceDim; ++j) {
            fscanf(DataFilePointer, "%lf", &Action[i*ActionSpaceDim+j]);
        }        
        // Read Advantage
        fscanf(DataFilePointer, "%lf", &Advantage[i]);        
    }
    
    // Close Data File
    fclose(DataFilePointer);
    
    
    //////////////////// Main Loop Over All Samples ////////////////////

    // Measure Elapsed Time
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);

    for (size_t iter=0; iter<NumSamples; iter++) {
    
        //////////////////// Combined Forward Propagation ////////////////////
    
        // Initialise the Input Layer
        for (size_t i=0; i<ObservSpaceDim; ++i) {
              Layer[0][i] = Observ[iter*ObservSpaceDim+i];
            RxLayer[0][i] = 0;
            RyLayer[0][i] = 0;
        }
    
        // Forward Propagation
        for (size_t i=0; i<NumLayers-1; ++i) {

            size_t CurrLayerSize = LayerSize[i];
            size_t NextLayerSize = LayerSize[i+1];
            size_t j, k;
            
            // Propagate from Layer[i] to Layer[i+1]
            #pragma omp parallel for private(j,k) shared(Layer, RxLayer, RyLayer, W, VW, B, VB, AcFunc) schedule(static)
            for (j=0; j<NextLayerSize; ++j) {
                
                // Initialise x_j and R{x_j} in next layer
                // Here we just use y_j's memory space to store x_j temoporarily
                  Layer[i+1][j] = B[i][j];
                RxLayer[i+1][j] = VB[i][j];
                
                for (k=0; k<CurrLayerSize; ++k) {
                    // From Neuron #k in Layer[i] to Neuron #j in Layer[i+1]
                      Layer[i+1][j] +=   Layer[i][k] *  W[i][k*NextLayerSize+j];
                    RxLayer[i+1][j] += RyLayer[i][k] *  W[i][k*NextLayerSize+j];
                    RxLayer[i+1][j] +=   Layer[i][k] * VW[i][k*NextLayerSize+j];
                }

                // Calculate y_j and R{y_j} in next layer. Note that R{y_j} depends on y_j
                switch (AcFunc[i+1]) {
                    // Linear Activation Function: Ac(x) = (x)
                    case 'l': {
                        RyLayer[i+1][j] = RxLayer[i+1][j];
                        break;
                    }
                    // tanh() Activation Function
                    case 't': {
                          Layer[i+1][j] = tanh(Layer[i+1][j]);
                        RyLayer[i+1][j] = RxLayer[i+1][j] * (1 - Layer[i+1][j] * Layer[i+1][j]);
                        break;
                    }
                    // 0.1x Activation Function
                    case 'o': {
                          Layer[i+1][j] = 0.1 *   Layer[i+1][j];
                        RyLayer[i+1][j] = 0.1 * RxLayer[i+1][j];
                        break;
                    }
                    // sigmoid Activation Function
                    case 's': {
                          Layer[i+1][j] = 1.0 / ( 1 + exp(-Layer[i+1][j]) );
                        RyLayer[i+1][j] = RxLayer[i+1][j] * Layer[i+1][j] * (1 - Layer[i+1][j]);
                        break;
                    }
                    // Default: Activation Function not supported
                    default: {
                        printf("[ERROR] AC Function for Layer[%zu] is %c. Unsupported.\n", i+1, AcFunc[i+1]);
                    }
                }
            }
        }

        // Check whether the forward propagation output is correct
        for (size_t i=0; i<ActionSpaceDim; ++i) {
            double output   = Layer[NumLayers-1][i];
            double expected = Mean[iter*ActionSpaceDim+i];
            double err      = fabs( (output - expected) / expected ) * 100;
            if (err>1) printf("out[%zu] = %e, mean = %e => %.4f%% Difference\n", i, output, expected, err);
        }


        //////////////////// Pearlmutter Backward Propagation ////////////////////


        // Gradient Initialisation
        // Calculating R{} Gradient of KL w.r.t. output values from the final layer, i.e. R{d(KL)/d(mean_i)}
        for (size_t i=0; i<ActionSpaceDim; ++i) {
            RGLayer[NumLayers-1][i] = RyLayer[NumLayers-1][i] / Std[i] / Std[i];
        }

        // Backward Propagation
        for (size_t i=NumLayers-1; i>0; --i) {
            
            size_t CurrLayerSize = LayerSize[i];
            size_t PrevLayerSize = LayerSize[i-1];
            size_t j, k;

            // Propagate from Layer[i] to Layer[i-1]
            #pragma omp parallel for private(j) shared(Layer, RGLayer, RGB) schedule(static)            
            for (j=0; j<CurrLayerSize; ++j) {

                // Calculating R{} Gradient of KL w.r.t. pre-activated values in Layer[i], i.e. R{d(KL)/d(x_i)}
                // Differentiate the activation function
                switch (AcFunc[i]) {
                    // Linear Activation Function: Ac(x) = (x)
                    case 'l': {break;}
                    // tanh() Activation Function: tanh' = 1 - tanh^2
                    case 't': {RGLayer[i][j] = (1-Layer[i][j]*Layer[i][j])*RGLayer[i][j]; break;}
                    // 0.1x Activation Function
                    case 'o': {RGLayer[i][j] = 0.1 * RGLayer[i][j]; break;}
                    // sigmoid Activation Function: sigmoid' = sigmoid * (1 - sigmoid)
                    case 's': {RGLayer[i][j] = RGLayer[i][j]*Layer[i][j]*(1-Layer[i][j]); break;}
                    // Default: Activation Function not supported
                    default: {
                        fprintf(stderr, "[ERROR] AC Function for Layer [%zu] is %c. Unsupported.\n", i, AcFunc[i]);
                    }
                }

                // The R{} derivative w.r.t to Bias is the same as that w.r.t. the pre-activated value
                RGB[i-1][j] = RGLayer[i][j];
            }

            // Calculate the R{} derivative w.r.t. to Weight and the output values from Layer[i]
            #pragma omp parallel for private(j,k) shared(Layer, RGLayer, W, RGW) schedule(static)
            for (j=0; j<PrevLayerSize; ++j) {
                double temp = 0;
                for (k=0; k<CurrLayerSize; ++k) {
                    // The R{} Derivative w.r.t. to the weight from Neuron #j in Layer[i-1] to Neuron #k in Layer[i]
                    RGW[i-1][j*CurrLayerSize+k] = Layer[i-1][j] * RGLayer[i][k];
                    // Accumulate the Gradient from Neuron #k in Layer[i] to Neuron #j in Layer[i-1]
                    temp += W[i-1][j*CurrLayerSize+k] * RGLayer[i][k];
                }
                RGLayer[i-1][j] = temp;
            }         
        }
        
        // Accumulate the Fisher-Vector Product to result
        pos = 0;
        for (size_t i=0; i<NumLayers-1; ++i) {
            size_t curLayerDim = LayerSize[i];
            size_t nextLayerDim = LayerSize[i+1];
            for (size_t j=0; j<curLayerDim;++j) {
                for (size_t k=0; k<nextLayerDim; ++k) {
                    Result[pos] += RGW[i][j*nextLayerDim+k];
                    pos++;
                }
            }
            for (size_t k=0; k<nextLayerDim; ++k) {
                Result[pos] += RGB[i][k];
                pos++;
            }
        }
        for (size_t k=0; k<ActionSpaceDim; ++k) {
            Result[pos] += 2 * VLogStd[k];
            pos++;
        }

    
    } // End of iteration over current sample


    // Averaging Fisher Vector Product over the samples and apply CG Damping
    #pragma omp parallel for
    for (size_t i=0; i<pos; ++i) {
        Result[i] = Result[i] / (double)NumSamples + CG_Damping * Input[i];
    }

    gettimeofday(&tv2, NULL);
    double runtimeS = ((tv2.tv_sec-tv1.tv_sec) * (double)1E6 + (tv2.tv_usec-tv1.tv_usec)) / (double)1E6;

    //////////////////// Clean Up ////////////////////

    // clean up
    for (size_t i=0; i<NumLayers; ++i) {
        free(Layer[i]); free(RxLayer[i]); free(RyLayer[i]); free(RGLayer[i]);
    }
    for (size_t i=0; i<NumLayers-1; ++i) {
        free(W[i]); free(VW[i]); free(RGW[i]);
        free(B[i]); free(VB[i]); free(RGB[i]);
    }
    free(Observ); free(Mean); free(Std); free(Action); free(Advantage); free(VLogStd);

    return runtimeS;



}
