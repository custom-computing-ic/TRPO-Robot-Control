#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "omp.h"

#include "TRPO.h"


double TRPO(TRPOparam param, double *Result, size_t NumThreads) {

    //////////////////// Remarks ////////////////////

    // Result: Updated Policy Parameters

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
    double ResidualTh       = 1e-10;
    size_t MaxIter          = 10;
    double MaxKL            = 0.01;
    double MaxBackTracks    = 10;
    double AcceptRatio      = 0.1;



    // Dimension of Observation Space
    const size_t ObservSpaceDim = LayerSize[0];
    
    // Dimension of Action Space
    const size_t ActionSpaceDim = LayerSize[NumLayers-1];

    // Number of Policy Parameters
    size_t NumParams = NumParamsCalc(param.LayerSize, param.NumLayers);

    // iterator when traversing through input vector and result vector
    size_t pos;


    //////////////////// Memory Allocation - Model ////////////////////
    
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
    
    // LogStd[i] is the log of std[i] in the policy
    double * LogStd = (double *) calloc(ActionSpaceDim, sizeof(double));


    //////////////////// Memory Allocation - Policy Gradient ////////////////////

    // The Policy Gradient Vector (PG) is the gradient of Surrogate Loss w.r.t. to policy parameters
    // -PG is the input to the Conjugate Gradient (CG) function
    // There is one-to-one correspondence between PG and policy parameters (W and B of neural network, LogStd)
    double * PGW [NumLayers-1];
    double * PGB [NumLayers-1];
    for (size_t i=0; i<NumLayers-1; ++i) {
        PGW[i] = (double *) calloc(LayerSize[i]*LayerSize[i+1], sizeof(double));
        PGB[i] = (double *) calloc(LayerSize[i+1], sizeof(double));
    }
    
    // Allocate Memory for Policy Gradient corresponding to LogStd
    double * PGLogStd = (double *) calloc(ActionSpaceDim, sizeof(double));


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
    
    
    //////////////////// Memory Allocation - Ordinary Forward and Backward Propagation ////////////////////

    // Layer[i] : Memory of each layer's outputs, i.e. y_i
    // GLayer[i]: Gradient of Loss Function w.r.t. the pre-activation values in Layer[i], i.e. d(Loss)/d(x_i)
    double * Layer  [NumLayers];
    double * GLayer [NumLayers];
    for (size_t i=0; i<NumLayers; ++i) {
        Layer[i]  = (double *) calloc(LayerSize[i], sizeof(double));
        GLayer[i] = (double *) calloc(LayerSize[i], sizeof(double));
    }

    // GW[i]: Gradient of Loss Function w.r.t to Neural Network Weight W[i]
    // GB[i]: Gradient of Loss Function w.r.t to Neural Network Bias B[i]
    // There is one-to-one correspondence between: GW[i] and W[i], GB[i] and B[i], GStd[i] and Std[i]
    double * GW [NumLayers-1];
    double * GB [NumLayers-1];
    for (size_t i=0; i<NumLayers-1; ++i) {
        GW[i] = (double *) calloc(LayerSize[i]*LayerSize[i+1], sizeof(double));
        GB[i] = (double *) calloc(LayerSize[i+1], sizeof(double));
    }

    // GLogStd[i]: Gradient of Loss Function w.r.t LogStd[i]
    double * GLogStd = (double *) calloc(ActionSpaceDim, sizeof(double));


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

    // RGLogStd[i]: R{} Gradient of KL w.r.t LogStd[i]
    double * RGLogStd = (double *) calloc(ActionSpaceDim, sizeof(double));


    //////////////////// Memory Allocation - Conjugate Gradient (CG) ////////////////////

    // These names correspond to the names in the TRPO Python code
    double * b = (double *) calloc(NumParams, sizeof(double));
    double * p = (double *) calloc(NumParams, sizeof(double));
    double * r = (double *) calloc(NumParams, sizeof(double));
    double * x = (double *) calloc(NumParams, sizeof(double));
    double * z = (double *) calloc(NumParams, sizeof(double));
    
    
    //////////////////// Memory Allocation - Line Search ////////////////////

    // These names correspond to the names in the TRPO Python code
    // Note: In Line Search we also need a vector called x
    //       Here we just use the x declared for Conjugate Gradient for simlicity
    //       The x used in Line Search has nothing to do with the x used in CG
    //       They just have the same type and size
    double * fullstep = (double *) calloc(NumParams, sizeof(double));
    double * theta    = (double *) calloc(NumParams, sizeof(double));
    double * xnew     = (double *) calloc(NumParams, sizeof(double));


    //////////////////// Load Model ////////////////////
    
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
    for (size_t k=0; k<ActionSpaceDim; ++k) {
        fscanf(ModelFilePointer, "%lf", &LogStd[k]);
    }

    // Close Model File
    fclose(ModelFilePointer);
   
    
    //////////////////// Load Simulation Data ////////////////////
    
    // Open Data File that contains Mean, std and Observation
    FILE *DataFilePointer = fopen(DataFile, "r");
    if (DataFilePointer==NULL) {
        fprintf(stderr, "[ERROR] Cannot open Data File [%s]. \n", DataFile);
        return -1;
    }
    
    // Read Mean, Std and Observation - Note that Std = exp(LogStd)
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
    
    
    //////////////////// Main Computation Begins ////////////////////

    // Measure Elapsed Time
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);


    //////////////////// Computing Policy Gradient ////////////////////

    for (size_t iter=0; iter<NumSamples; iter++) {
    
        ///////// Ordinary Forward Propagation /////////
    
        // Assign Input Values
        for (size_t i=0; i<ObservSpaceDim; ++i) Layer[0][i] = Observ[iter*ObservSpaceDim+i];
    
        // Forward Propagation
        for (size_t i=0; i<NumLayers-1; ++i) {
            
            // Propagate from Layer[i] to Layer[i+1]
            for (size_t j=0; j<LayerSize[i+1]; ++j) {
                
                // Calculating pre-activated value for item[j] in next layer
                Layer[i+1][j] = B[i][j];
                for (size_t k=0; k<LayerSize[i]; ++k) {
                    // From Neuron #k in Layer[i] to Neuron #j in Layer[i+1]
                    Layer[i+1][j] += Layer[i][k] * W[i][k*LayerSize[i+1]+j];
                }
            
                // Apply Activation Function
                switch (AcFunc[i+1]) {
                    // Linear Activation Function: Ac(x) = (x)
                    case 'l': {break;}
                    // tanh() Activation Function
                    case 't': {Layer[i+1][j] = tanh(Layer[i+1][j]); break;}
                    // 0.1x Activation Function
                    case 'o': {Layer[i+1][j] = 0.1*Layer[i+1][j]; break;}
                    // sigmoid Activation Function
                    case 's': {Layer[i+1][j] = 1.0/(1+exp(-Layer[i+1][j])); break;}
                    // Default: Activation Function not supported
                    default: {
                        printf("[ERROR] Activation Function for Layer [%zu] is %c. Unsupported.\n", i+1, AcFunc[i+1]);
                        return -1;
                    }
                }
            }
        }
        
        ///////// Ordinary Backward Propagation /////////         

        // Gradient Initialisation
        // Assign the derivative of Surrogate Loss w.r.t. Mean (output values from the final layer) and LogStd
        for (size_t i=0; i<ActionSpaceDim; ++i) {
            double temp = (Action[iter*ActionSpaceDim+i] - Mean[iter*ActionSpaceDim+i]) / exp(LogStd[i]);
            GLayer[NumLayers-1][i] = Advantage[iter] * temp / exp(LogStd[i]);
            GLogStd[i] = Advantage[iter] * (temp * temp - 1);
        }

        // Backward Propagation
        for (size_t i=NumLayers-1; i>0; --i) {
       
            // Propagate from Layer[i] to Layer[i-1]
            for (size_t j=0; j<LayerSize[i]; ++j) {

                // Differentiate the activation function
                switch (AcFunc[i]) {
                    // Linear Activation Function: Ac(x) = (x)
                    case 'l': {break;}
                    // tanh() Activation Function: tanh' = 1 - tanh^2
                    case 't': {GLayer[i][j] = GLayer[i][j] * (1- Layer[i][j] * Layer[i][j]); break;}
                    // 0.1x Activation Function
                    case 'o': {GLayer[i][j] = 0.1 * GLayer[i][j]; break;}
                    // sigmoid Activation Function: sigmoid' = sigmoid * (1 - sigmoid)
                    case 's': {GLayer[i][j] = GLayer[i][j] * Layer[i][j] * (1- Layer[i][j]); break;}
                    // Default: Activation Function not supported
                    default: {
                        fprintf(stderr, "[ERROR] Activation Function for Layer[%zu] is %c. Unsupported.\n", i, AcFunc[i]);
                        return -1;
                    }
                }
                
                // The derivative w.r.t to Bias is the same as that w.r.t. the pre-activated value
                GB[i-1][j] = GLayer[i][j];
            }
        
            // Calculate the derivative w.r.t. to Weight
            for (size_t j=0; j<LayerSize[i-1]; ++j) {
                for (size_t k=0; k<LayerSize[i]; ++k) {
                    // The Derivative w.r.t. to the weight from Neuron #j in Layer[i-1] to Neuron #k in Layer[i]
                    GW[i-1][j*LayerSize[i]+k] = GLayer[i][k] * Layer[i-1][j];
                }
            }
        
            // Calculate the derivative w.r.t. the output values from Layer[i]
            for (size_t j=0; j<LayerSize[i-1]; ++j) {
                GLayer[i-1][j] = 0;
                for (size_t k=0; k<LayerSize[i]; ++k) {
                    // Accumulate the Gradient from Neuron #k in Layer[i] to Neuron #j in Layer[i-1]
                    GLayer[i-1][j] += GLayer[i][k] * W[i-1][j*LayerSize[i]+k];
                }
            }
        
        }

        
        // Accumulate the Policy Gradient to b
        pos = 0;
        for (size_t i=0; i<NumLayers-1; ++i) {
            size_t curLayerDim = LayerSize[i];
            size_t nextLayerDim = LayerSize[i+1];
            for (size_t j=0; j<curLayerDim;++j) {
                for (size_t k=0; k<nextLayerDim; ++k) {
                    b[pos] += GW[i][j*nextLayerDim+k];
                    pos++;
                }
            }
            for (size_t k=0; k<nextLayerDim; ++k) {
                b[pos] += GB[i][k];
                pos++;
            }
        }
        for (size_t k=0; k<ActionSpaceDim; ++k) {
            b[pos] += GLogStd[k];
            pos++;
        }
    
    } // End of iteration over current sample
    
    // Averaging Policy Gradient over the samples - Policy Gradient is held in b
    // Note this corresponds to -g in the Python code: b = -g
    #pragma omp parallel for
    for (size_t i=0; i<pos; ++i) {
        b[i] = b[i] / (double)NumSamples;
    }


    //////////////////// Computing Search Direction ////////////////////

    ///////// Conjugate Gradient /////////

    // This function implements Conjugate Gradient algorithm to solve linear equation Ax=b
    //     x: The Conjugate Gradient Result, i.e. solution x to Ax=b
    //        In TRPO context, x is the Step Direction of the line search (stepdir in the Python code)
    //     b: Vector b in the equation Ax=b

    // Initialisation
    double rdotr = 0;
    for (size_t i=0; i<NumParams; ++i) {
        p[i] = b[i];
        r[i] = b[i];
        rdotr += r[i] * r[i];
    }
    
    // Iterative Solver    
    for (size_t it=0; it<=MaxIter; ++it) {

        // Calculate Frobenius Norm of x
        double FrobNorm = 0;

        #pragma omp parallel for reduction (+:FrobNorm)
        for (size_t i=0; i<NumParams; ++i) {
            FrobNorm += x[i] * x[i];
        }
        FrobNorm = sqrt(FrobNorm);

        printf("CG Iter[%zu] Residual Norm=%.12e, Soln Norm=%.12e\n", it, rdotr, FrobNorm);
        
        // Check Termination Condition
        if (rdotr<ResidualTh || it==MaxIter) {
            for (size_t i=0; i<NumParams; ++i) z[i] = x[i];
            break;
        }
        
        ///////// Fisher Vector Product Computation z = FVP(p) /////////
        
        // Init PGW, PGB, PGLogStd from p
        // Init z to 0
        pos = 0;
        for (size_t i=0; i<NumLayers-1; ++i) {
            size_t curLayerDim  = LayerSize[i];
            size_t nextLayerDim = LayerSize[i+1];
            for (size_t j=0; j<curLayerDim;++j) {
                for (size_t k=0; k<nextLayerDim; ++k) {
                    PGW[i][j*nextLayerDim+k] = p[pos];
                    z[pos] = 0;
                    pos++;
                }
            }
            for (size_t k=0; k<nextLayerDim; ++k) {
                PGB[i][k] = p[pos];
                z[pos] = 0;
                pos++;
            }
        }
        for (size_t k=0; k<ActionSpaceDim; ++k) {
            PGLogStd[k] = p[pos];
            z[pos] = 0;
            pos++;
        }
        
        for (size_t iter=0; iter<NumSamples; iter++) {
    
            ///////// Combined Forward Propagation /////////
    
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
                #pragma omp parallel for private(j,k) shared(Layer, RxLayer, RyLayer, W, PGW, B, PGB, AcFunc) schedule(static)
                for (j=0; j<NextLayerSize; ++j) {
                
                    // Initialise x_j and R{x_j} in next layer
                    // Here we just use y_j's memory space to store x_j temoporarily
                      Layer[i+1][j] = B[i][j];
                    RxLayer[i+1][j] = PGB[i][j];
                
                    for (k=0; k<CurrLayerSize; ++k) {
                        // From Neuron #k in Layer[i] to Neuron #j in Layer[i+1]
                          Layer[i+1][j] +=   Layer[i][k] *   W[i][k*NextLayerSize+j];
                        RxLayer[i+1][j] += RyLayer[i][k] *   W[i][k*NextLayerSize+j];
                        RxLayer[i+1][j] +=   Layer[i][k] * PGW[i][k*NextLayerSize+j];
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


            ///////// Pearlmutter Backward Propagation /////////

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
        
            // Accumulate the Fisher-Vector Product to z
            pos = 0;
            for (size_t i=0; i<NumLayers-1; ++i) {
                size_t curLayerDim = LayerSize[i];
                size_t nextLayerDim = LayerSize[i+1];
                for (size_t j=0; j<curLayerDim;++j) {
                    for (size_t k=0; k<nextLayerDim; ++k) {
                        z[pos] += RGW[i][j*nextLayerDim+k];
                        pos++;
                    }
                }
                for (size_t k=0; k<nextLayerDim; ++k) {
                    z[pos] += RGB[i][k];
                    pos++;
                }
            }
            for (size_t k=0; k<ActionSpaceDim; ++k) {
                z[pos] += 2 * PGLogStd[k];
                pos++;
            }
            
        } // End of iteration over current sample


        // Averaging Fisher Vector Product over the samples and apply CG Damping
        #pragma omp parallel for
        for (size_t i=0; i<pos; ++i) {
            z[i] = z[i] / (double)NumSamples + CG_Damping * p[i];
        }
    
        //////////////// FVP Finish        
    
        // Update x and r
        double pdotz = 0;

        #pragma omp parallel for reduction (+:pdotz)
        for (size_t i=0; i<NumParams; ++i) {
            pdotz += p[i] * z[i];
        }
        double v = rdotr / pdotz;
        #pragma omp parallel for
        for (size_t i=0; i<NumParams; ++i) {
            x[i] += v * p[i];
            r[i] -= v * z[i];
        }
        
        // Update p
        double newrdotr = 0;
        #pragma omp parallel for reduction (+:newrdotr)
        for (size_t i=0; i<NumParams; ++i) {
            newrdotr += r[i] * r[i];
        }
        double mu = newrdotr / rdotr;
        #pragma omp parallel for
        for (size_t i=0; i<NumParams; ++i) {
            p[i] = r[i] + mu * p[i];
        }
        
        // Update rdotr
        rdotr = newrdotr;

    }


    // Calculate another Fisher Vector Product - code reuse opportunity
    
    ///////// Fisher Vector Product Computation z = FVP(x) /////////
    
    // Init PGW, PGB, PGLogStd from x
    // Init z to 0
    pos = 0;
    for (size_t i=0; i<NumLayers-1; ++i) {
        size_t curLayerDim  = LayerSize[i];
        size_t nextLayerDim = LayerSize[i+1];
        for (size_t j=0; j<curLayerDim;++j) {
            for (size_t k=0; k<nextLayerDim; ++k) {
                PGW[i][j*nextLayerDim+k] = x[pos];
                z[pos] = 0;
                pos++;
            }
        }
    for (size_t k=0; k<nextLayerDim; ++k) {
            PGB[i][k] = x[pos];
            z[pos] = 0;
            pos++;
        }
    }
    for (size_t k=0; k<ActionSpaceDim; ++k) {
        PGLogStd[k] = x[pos];
        z[pos] = 0;
        pos++;
    }
        
    for (size_t iter=0; iter<NumSamples; iter++) {
    
        ///////// Combined Forward Propagation /////////
    
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
            #pragma omp parallel for private(j,k) shared(Layer, RxLayer, RyLayer, W, PGW, B, PGB, AcFunc) schedule(static)
            for (j=0; j<NextLayerSize; ++j) {
                
                // Initialise x_j and R{x_j} in next layer
                // Here we just use y_j's memory space to store x_j temoporarily
                  Layer[i+1][j] = B[i][j];
                RxLayer[i+1][j] = PGB[i][j];
                
                for (k=0; k<CurrLayerSize; ++k) {
                    // From Neuron #k in Layer[i] to Neuron #j in Layer[i+1]
                      Layer[i+1][j] +=   Layer[i][k] *   W[i][k*NextLayerSize+j];
                    RxLayer[i+1][j] += RyLayer[i][k] *   W[i][k*NextLayerSize+j];
                    RxLayer[i+1][j] +=   Layer[i][k] * PGW[i][k*NextLayerSize+j];
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


        ///////// Pearlmutter Backward Propagation /////////

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
        
        // Accumulate the Fisher-Vector Product to z
        pos = 0;
        for (size_t i=0; i<NumLayers-1; ++i) {
            size_t curLayerDim = LayerSize[i];
            size_t nextLayerDim = LayerSize[i+1];
            for (size_t j=0; j<curLayerDim;++j) {
                for (size_t k=0; k<nextLayerDim; ++k) {
                    z[pos] += RGW[i][j*nextLayerDim+k];
                    pos++;
                }
            }
            for (size_t k=0; k<nextLayerDim; ++k) {
                z[pos] += RGB[i][k];
                pos++;
            }
        }
        for (size_t k=0; k<ActionSpaceDim; ++k) {
            z[pos] += 2 * PGLogStd[k];
            pos++;
        }
            
    } // End of iteration over current sample


    // Averaging Fisher Vector Product over the samples and apply CG Damping
    #pragma omp parallel for
    for (size_t i=0; i<pos; ++i) {
        z[i] = z[i] / (double)NumSamples + CG_Damping * x[i];
    }
    
    // Now z holds the Fisher Vector Product, x holds stepdir
    double shs = 0;
    #pragma omp parallel for reduction (+:shs)
    for (size_t i=0; i<NumParams; ++i) {
        shs += z[i] * x[i];
    }
    shs = shs * 0.5;
    printf("shs: %.14f\n", shs);
    
    
    // Lagrange Multiplier (lm in Python code)
    double lm = sqrt(shs / MaxKL);
    
    // Compute the 2-norm of the Policy Gradient
    double gnorm = 0;
    for (size_t i=0; i<NumParams; ++i) {
        gnorm += b[i] * b[i];
    }
    gnorm = sqrt(gnorm);
    
    printf("lagrange multiplier: %.14f, gnorm: %.14f\n", lm, gnorm);
    
    // Full Step
    #pragma omp parallel for
    for (size_t i=0; i<NumParams; ++i) {
        fullstep[i] = x[i] / lm;
    }
    
    // Inner product of Negative Policy Gradient -g and Step Direction
    double neggdotstepdir = 0;
    #pragma omp parallel for reduction (+:neggdotstepdir)
    for (size_t i=0; i<NumParams; ++i) {
        neggdotstepdir += b[i] * x[i];
    }


    //////////////////// Line Search ////////////////////
    
    // Init theta to x
    // If Line Search is unsuccessful, theta remains as x
    for (size_t i=0; i<NumParams; ++i) theta[i] = x[i];
    
    // Expected Improve Rate Line Search = slope dy/dx at initial point
    double expected_improve_rate = neggdotstepdir / lm;
    
    // Temporarily Save the Model Parameters in x
    // The x refers to the x in linesearch function in Python code
    // Note: Although the name is the same, the x here has nothing to do with the x in Conjugate Gradient
    
    // Copy the Model Parameters to x
    pos = 0;
    for (size_t i=0; i<NumLayers-1; ++i) {
        size_t curLayerDim = LayerSize[i];
        size_t nextLayerDim = LayerSize[i+1];
        for (size_t j=0; j<curLayerDim;++j) {
            for (size_t k=0; k<nextLayerDim; ++k) {
                x[pos] = W[i][j*nextLayerDim+k];
                pos++;
            }
        }
        for (size_t k=0; k<nextLayerDim; ++k) {
            x[pos] = B[i][k];
            pos++;
        }
    }
    for (size_t k=0; k<ActionSpaceDim; ++k) {
        x[pos] = LogStd[k];
        pos++;
    }    

    // Surrogate Loss of the current Model parameters = -Avg(Advantage)
    double fval = 0;
    #pragma omp parallel for reduction (+:fval)
    for (size_t i=0; i<NumSamples; ++i) {
        fval += Advantage[i];
    }    
    fval = -fval / (double) NumSamples;

    printf("fval before %.14e\n", fval);
    
    // Backtracking Line Search
    for (size_t i=0; i<MaxBackTracks; ++i) {
    
        // Step Fraction
        double stepfrac = pow(0.5, (double)i);
    
        // x New
        #pragma omp parallel for
        for (size_t i=0; i<NumParams; ++i) {
            xnew[i] = x[i] + stepfrac * fullstep[i];
        }
        
        ///////// Compute Surrogate Loss /////////
        
        // Init W, B, LogStd from xnew
        pos = 0;
        for (size_t i=0; i<NumLayers-1; ++i) {
            size_t curLayerDim  = LayerSize[i];
            size_t nextLayerDim = LayerSize[i+1];
            for (size_t j=0; j<curLayerDim;++j) {
                for (size_t k=0; k<nextLayerDim; ++k) {
                    W[i][j*nextLayerDim+k] = xnew[pos];
                    pos++;
                }
            }
            for (size_t k=0; k<nextLayerDim; ++k) {
                B[i][k] = xnew[pos];
                pos++;
            }
        }
        for (size_t k=0; k<ActionSpaceDim; ++k) {
            LogStd[k] = xnew[pos];
            pos++;
        }
        
        // Init Surrogate Loss to 0
        double surr = 0;
        
        for (size_t iter=0; iter<NumSamples; iter++) {
    
            ///////// Ordinary Forward Propagation /////////
    
            // Assign Input Values
            for (size_t i=0; i<ObservSpaceDim; ++i) Layer[0][i] = Observ[iter*ObservSpaceDim+i];
    
            // Forward Propagation
            for (size_t i=0; i<NumLayers-1; ++i) {
            
                // Propagate from Layer[i] to Layer[i+1]
                for (size_t j=0; j<LayerSize[i+1]; ++j) {
                
                    // Calculating pre-activated value for item[j] in next layer
                    Layer[i+1][j] = B[i][j];
                    for (size_t k=0; k<LayerSize[i]; ++k) {
                        // From Neuron #k in Layer[i] to Neuron #j in Layer[i+1]
                        Layer[i+1][j] += Layer[i][k] * W[i][k*LayerSize[i+1]+j];
                    }
            
                    // Apply Activation Function
                    switch (AcFunc[i+1]) {
                        // Linear Activation Function: Ac(x) = (x)
                        case 'l': {break;}
                        // tanh() Activation Function
                        case 't': {Layer[i+1][j] = tanh(Layer[i+1][j]); break;}
                        // 0.1x Activation Function
                        case 'o': {Layer[i+1][j] = 0.1*Layer[i+1][j]; break;}
                        // sigmoid Activation Function
                        case 's': {Layer[i+1][j] = 1.0/(1+exp(-Layer[i+1][j])); break;}
                        // Default: Activation Function not supported
                        default: {
                            printf("[ERROR] Activation Function for Layer [%zu] is %c. Unsupported.\n", i+1, AcFunc[i+1]);
                            return -1;
                        }
                    }
                }
            }

            // Surrogate Loss Calculation

            // LoglikelihoodDifference = logp_i - oldlogp_i
            // Here, logp_i is derived from xnew, oldlogp_i is derived from x (Mean in the simulation data)
            double LoglikelihoodDifference = 0;
            for (size_t i=0; i<ActionSpaceDim; ++i) {
                double temp_x    = (Action[iter*ActionSpaceDim+i] - Mean[iter*ActionSpaceDim+i]) / Std[i];
                double temp_xnew = (Action[iter*ActionSpaceDim+i] - Layer[NumLayers-1][i]) / exp(LogStd[i]);
                LoglikelihoodDifference += temp_x*temp_x - temp_xnew*temp_xnew + log(Std[i]) - LogStd[i];
            }
            LoglikelihoodDifference = LoglikelihoodDifference * 0.5;
            
            // Accumulate Surrogate Loss
            surr += exp(LoglikelihoodDifference) * Advantage[iter];
      
        }
        
        // Average Surrogate Loss over the samples to get newfval
        double newfval = -surr / (double) NumSamples;
        
        // Improvement in terms of Surrogate Loss
        double actual_improve = fval - newfval;
        
        // Expected Improvement
        double expected_improve = expected_improve_rate * stepfrac;
        
        // Improvement Ratio
        double ratio = actual_improve / expected_improve;
        
        printf("a/e/r %.14f / %.14f / %.14f\n", actual_improve, expected_improve, ratio);
        
        // Check breaking condition - has Line Search succeeded?
        if ( (ratio > AcceptRatio) && (actual_improve > 0) ) {
            // If Line Search is successful, update parameters and quit
            for (size_t i=0; i<NumParams; ++i) theta[i] = xnew[i];
            break;
        }
    
    }   // End of Line Search
    
    // Copy theta to Result
    // Note that these are the updated Model parameters
    for (size_t i=0; i<NumParams; ++i) Result[i] = theta[i];

    gettimeofday(&tv2, NULL);
    double runtimeS = ((tv2.tv_sec-tv1.tv_sec) * (double)1E6 + (tv2.tv_usec-tv1.tv_usec)) / (double)1E6;

    //////////////////// Clean Up ////////////////////

    // Model - From Model File
    for (size_t i=0; i<NumLayers-1; ++i) {
        free(W[i]); free(B[i]);
    }
    free(LogStd);

    // Simulation Data - From Data File
    free(Observ); free(Mean); free(Std); free(Action); free(Advantage);
    
    // Forward and Backward Propagation
    for (size_t i=0; i<NumLayers; ++i) {
        // Ordinary Forward and Backward Propagation
        free(Layer[i]); free(GLayer[i]);
        // Pearlmutter Forward and Backward Propagation
        free(RxLayer[i]); free(RyLayer[i]); free(RGLayer[i]);
    }
    
    // Gradient
    for (size_t i=0; i<NumLayers-1; ++i) {
        // Gradient - temporary storage
        free(GW[i]); free(GB[i]);    
        // Policy Gradient
        free(PGW[i]); free(PGB[i]);
        // Pearlmutter R{} Gradient
        free(RGW[i]); free(RGB[i]);
    }
    
    // Gradient - LogStd
    free(GLogStd); free(PGLogStd); free(RGLogStd);

    // Conjugate Gradient
    free(b); free(p); free(r); free(x); free(z);

    // Line Search
    free(fullstep); free(xnew); free(theta);

    return runtimeS;

}
