#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "omp.h"

#include "TRPO.h"
#include "lbfgs.h"


/**
 * Callback interface to provide objective function and gradient evaluations.
 *
 *  The lbfgs() function call this function to obtain the values of objective
 *  function and its gradients when needed. A client program must implement
 *  this function to evaluate the values of the objective function and its
 *  gradients, given current values of variables.
 *  
 *  @param  param    The user data sent for lbfgs() function by the client.
 *  @param  x           The current values of variables.
 *  @param  g           The gradient vector. The callback function must compute
 *                      the gradient values for the current variables.
 *  @param  n           The number of variables.
 *  @param  step        The current step of the line search routine.
 *  @retval lbfgsfloatval_t The value of the objective function for the current
 *                          variables.
 */
lbfgsfloatval_t evaluate(void *param_in, const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n, const lbfgsfloatval_t step) {

    // Note that the number of parameters in this implementation must be a multiple of 16
    // Thus there are some padding zeros in the end

    /////////// Parameters and Pointers ///////////

    // Cast void* to TRPOBaselineParam*
    TRPOBaselineParam* param = (TRPOBaselineParam*) param_in;

    // Read Paramaters and Memory Addresses from param
    const size_t NumLayers      = param -> NumLayers;
    const size_t ObservSpaceDim = param -> ObservSpaceDim;
    const size_t NumEpBatch     = param -> NumEpBatch;
    const size_t EpLen          = param -> EpLen;
    const size_t NumSamples     = param -> NumSamples;
    const size_t NumParams      = param -> NumParams;
    char * AcFunc               = param -> AcFunc;
    
    // For Forward Propagation and Back Propagation
    size_t * LayerSizeBase      = param -> LayerSizeBase;
    double ** WBase             = param -> WBase;
    double ** BBase             = param -> BBase;
    double ** LayerBase         = param -> LayerBase;
    double ** GWBase            = param -> GWBase;
    double ** GBBase            = param -> GBBase;
    double ** GLayerBase        = param -> GLayerBase;
    
    // Training Data
    double * Observ             = param -> Observ;
    double * Target             = param -> Target;
    double * Predict            = param -> Predict;


    /////////// Initialisation ///////////

    // Init Weight and Bias from x
    // Init Gradient to 0
    size_t pos = 0;
    for (size_t i=0; i<NumLayers-1; ++i) {
        size_t curLayerDim  = LayerSizeBase[i];
        size_t nextLayerDim = LayerSizeBase[i+1];
        for (size_t j=0; j<curLayerDim;++j) {
            for (size_t k=0; k<nextLayerDim; ++k) {
                WBase[i][j*nextLayerDim+k] = x[pos];
                g[pos] = 0;
                pos++;
            }
        }
        for (size_t k=0; k<nextLayerDim; ++k) {
            BBase[i][k] = x[pos];
            g[pos] = 0;
            pos++;
        }
    }
    for (size_t i=pos; i<n; ++i) g[pos] = 0;


    /////////// Forward Propagation and Back Propagation ///////////
    
    // For each episode
    for (int ep=0; ep<NumEpBatch; ++ep) {

        // For each timestep in each episode
        for (int currentStep=0; currentStep<EpLen; ++currentStep) {
            
            // Obsevation Vector and Normalised Timestep
            pos = ep*EpLen + currentStep;
            for (int i=0; i<ObservSpaceDim; ++i) {
                LayerBase[0][i] = Observ[pos*ObservSpaceDim+i];
            }
            LayerBase[0][ObservSpaceDim] = (double) currentStep / (double) EpLen;

            
            /////////// Forward Propagation ///////////
            
            for (size_t i=0; i<NumLayers-1; ++i) {
            
                // Propagate from Layer[i] to Layer[i+1]
                for (size_t j=0; j<LayerSizeBase[i+1]; ++j) {
                
                    // Calculating pre-activated value for item[j] in next layer
                    LayerBase[i+1][j] = BBase[i][j];
                    for (size_t k=0; k<LayerSizeBase[i]; ++k) {
                        // From Neuron #k in Layer[i] to Neuron #j in Layer[i+1]
                        LayerBase[i+1][j] += LayerBase[i][k] * WBase[i][k*LayerSizeBase[i+1]+j];
                    }
            
                    // Apply Activation Function
                    switch (AcFunc[i+1]) {
                        // Linear Activation Function: Ac(x) = (x)
                        case 'l': {break;}
                        // tanh() Activation Function
                        case 't': {LayerBase[i+1][j] = tanh(LayerBase[i+1][j]); break;}
                        // Default: Activation Function not supported
                        default: {
                            printf("[ERROR] Activation Function for Layer [%zu] is %c. Unsupported.\n", i+1, AcFunc[i+1]);
                            return -1;
                        }
                    }
                }
            }
                
            // Write result to Baseline
            Predict[pos] = LayerBase[NumLayers-1][0];

            
            /////////// Back Propagation ///////////

            // Gradient Initialisation - just MSE
            GLayerBase[NumLayers-1][0] = 2*(Predict[pos]-Target[pos]);   

            // Backward Propagation
            for (size_t i=NumLayers-1; i>0; --i) {
       
                // Propagate from Layer[i] to Layer[i-1]
                for (size_t j=0; j<LayerSizeBase[i]; ++j) {

                    // Differentiate the activation function
                    switch (AcFunc[i]) {
                        // Linear Activation Function: Ac(x) = (x)
                        case 'l': {break;}
                        // tanh() Activation Function: tanh' = 1 - tanh^2
                        case 't': {GLayerBase[i][j] = GLayerBase[i][j] * (1- LayerBase[i][j] * LayerBase[i][j]); break;}
                        // Default: Activation Function not supported
                        default: {
                            fprintf(stderr, "[ERROR] Activation Function for Layer[%zu] is %c. Unsupported.\n", i, AcFunc[i]);
                            return -1;
                        }
                    }
                
                    // The derivative w.r.t to Bias is the same as that w.r.t. the pre-activated value
                    GBBase[i-1][j] = GLayerBase[i][j];
                }
        
                // Calculate the derivative w.r.t. to Weight
                for (size_t j=0; j<LayerSizeBase[i-1]; ++j) {
                    for (size_t k=0; k<LayerSizeBase[i]; ++k) {
                        // The Derivative w.r.t. to the weight from Neuron #j in Layer[i-1] to Neuron #k in Layer[i]
                        GWBase[i-1][j*LayerSizeBase[i]+k] = GLayerBase[i][k] * LayerBase[i-1][j];
                    }
                }
        
                // Calculate the derivative w.r.t. the output values from Layer[i]
                for (size_t j=0; j<LayerSizeBase[i-1]; ++j) {
                    GLayerBase[i-1][j] = 0;
                    for (size_t k=0; k<LayerSizeBase[i]; ++k) {
                        // Accumulate the Gradient from Neuron #k in Layer[i] to Neuron #j in Layer[i-1]
                        GLayerBase[i-1][j] += GLayerBase[i][k] * WBase[i-1][j*LayerSizeBase[i]+k];
                    }
                }
            }
            
            
            // Accumulate the Gradient to g
            pos = 0;
            for (size_t i=0; i<NumLayers-1; ++i) {
                size_t curLayerDim = LayerSizeBase[i];
                size_t nextLayerDim = LayerSizeBase[i+1];
                for (size_t j=0; j<curLayerDim;++j) {
                    for (size_t k=0; k<nextLayerDim; ++k) {
                        g[pos] += GWBase[i][j*nextLayerDim+k];
                        pos++;
                    }
                }
                for (size_t k=0; k<nextLayerDim; ++k) {
                    g[pos] += GBBase[i][k];
                    pos++;
                }
            }
        }
    }
    
    /////////// Calculate Gradient ///////////
    
    pos = 0;
    for (size_t i=0; i<NumLayers-1; ++i) {
        size_t curLayerDim = LayerSizeBase[i];
        size_t nextLayerDim = LayerSizeBase[i+1];
        for (size_t j=0; j<curLayerDim;++j) {
            for (size_t k=0; k<nextLayerDim; ++k) {
                g[pos] = g[pos] / (double) NumSamples + 0.002 * WBase[i][j*nextLayerDim+k];
                pos++;
            }
        }
        for (size_t k=0; k<nextLayerDim; ++k) {
            g[pos] = g[pos] / (double) NumSamples + 0.002 * BBase[i][k];
            pos++;
        }
    }


    /////////// Loss Evaluation ///////////
    
    // Calculate Objective Function Value - MSE
    double mse = 0;
    for (size_t i=0; i<NumSamples; ++i) {
        mse += (Predict[i] - Target[i]) * (Predict[i] - Target[i]);
    }
    mse = mse / (double) NumSamples;

    // Calculate Objective Function Value - L2    
    double L2 = 0;
    for (size_t i=0; i<NumParams; ++i) {
        L2 += x[i] * x[i];
    }
    
    // Calculate Objective Function Value
    lbfgsfloatval_t ObjValue = mse + 1e-3 * L2;

    return ObjValue;
}


/**
 * Callback interface to receive the progress of the optimization process.
 *
 *  The lbfgs() function call this function for each iteration. Implementing
 *  this function, a client program can store or display the current progress
 *  of the optimization process.
 *
 *  @param  instance    The user data sent for lbfgs() function by the client.
 *  @param  x           The current values of variables.
 *  @param  g           The current gradient values of variables.
 *  @param  fx          The current value of the objective function.
 *  @param  xnorm       The Euclidean norm of the variables.
 *  @param  gnorm       The Euclidean norm of the gradients.
 *  @param  step        The line-search step used for this iteration.
 *  @param  n           The number of variables.
 *  @param  k           The iteration count.
 *  @param  ls          The number of evaluations called for this iteration.
 *  @retval int         Zero to continue the optimization process. Returning a
 *                      non-zero value will cancel the optimization process.
 */
int progress( void *instance, const lbfgsfloatval_t *x, const lbfgsfloatval_t *g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n, int k, int ls) {

    printf("[INFO] Baseline Update Iter %2d: Loss = %f, xnorm = %f, gnorm = %f, step = %f\n", k, fx, xnorm, gnorm, step);
    
    return 0;
}
