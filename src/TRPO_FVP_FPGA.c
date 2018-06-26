#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "omp.h"

#include "TRPO.h"
#include "Maxfiles.h"
#include "MaxSLiCInterface.h"


double FVP_FPGA (TRPOparam param, double *Result, double *Input){

    //////////////////// Remarks ////////////////////

    // This function computes the Fisher-Vector Product using Pearlmutter Algorithm
    // This version is customised to the case that KL is used as loss function
    // Input: the vector to be multiplied with the Fisher Information Matrix
    // Result: the Fisher-Vector Product
    // Remarks: The length of Input and Result must be the number of all trainable parameters in the network

    // Step1: Combined forward propagation
    // Step3: Pearlmutter backward propagation


    //////////////////// Read Parameters ////////////////////

    // Assign Parameters - For CPU and FPGA
    const size_t NumLayers  = param.NumLayers;
    size_t * LayerSize      = param.LayerSize;
    const size_t NumSamples = param.NumSamples;
    char * ModelFile        = param.ModelFile;
    char * DataFile         = param.DataFile;
    const double CG_Damping = param.CG_Damping;
    
    // Assign Parameters - For FPGA Only
    size_t * PaddedLayerSize = param.PaddedLayerSize;
    size_t * NumBlocks       = param.NumBlocks;
    

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
                pos++;
            }
        }
        for (size_t k=0; k<nextLayerDim; ++k) {
            VB[i][k] = Input[pos];
            pos++;
        }
    }
    for (size_t k=0; k<ActionSpaceDim; ++k) {
        VLogStd[k] = Input[pos];
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
    
    
    //////////////////// FPGA - Initialisation ////////////////////

	// Load Maxfile and Engine
	fprintf(stderr, "[INFO] Initialising FPGA...\n");
	max_file_t*  maxfile = TRPO_init();
	max_engine_t* engine = max_load(maxfile, "*");

    fprintf(stderr, "[INFO] Loading Model and Simulation Data...\n");

    // Calculate BlockDim
    size_t * BlockDim = (size_t *) calloc(NumLayers, sizeof(size_t));
    for (int i=0; i<NumLayers; ++i) BlockDim[i] = PaddedLayerSize[i] / NumBlocks[i];

    // Length of Weight and VWeight Initialisation Vector
    int WeightInitVecLength = 0;
    for (size_t i=0; i<NumLayers-1; ++i) {
        WeightInitVecLength += 2 * BlockDim[i] * PaddedLayerSize[i+1];
    }

    // Length of Observation Vector
    // Remarks: DRAM Write requires data bit-size to be a multiple of 384bytes
    //          Namely, the number of items must be a multiple of 48
    size_t ObservVecLength = WeightInitVecLength + NumSamples*BlockDim[0];
    size_t ObservVecWidth  = NumBlocks[0];
    size_t ActualObservVecItems = ObservVecLength * ObservVecWidth;
    size_t PaddedObservVecItems = (size_t) 48 * ceil( (double)ActualObservVecItems/48 );
    fprintf(stderr, "[INFO] Observation Vector (%zu bytes) padded to %zu bytes\n", ActualObservVecItems*8, PaddedObservVecItems*8);
    double * Observation = (double *) calloc(PaddedObservVecItems, sizeof(double));
    
    // Feed Weight and VWeight into Observation
    size_t RowNum = 0;
    for (size_t ID=0; ID<NumLayers-1; ++ID) {
        // Parameters of current
        size_t   InBlockDim = BlockDim[ID];
        size_t  NumInBlocks = NumBlocks[ID];
        size_t  OutBlockDim = BlockDim[ID+1];
        size_t NumOutBlocks = NumBlocks[ID+1];
        size_t OutLayerSize = LayerSize[ID+1];
        // Feed Weight of Layer[ID]
        for (size_t Y=0; Y<NumOutBlocks; ++Y) {
            for (size_t addrX=0; addrX<InBlockDim; ++addrX) {
                for (size_t addrY=0; addrY<OutBlockDim; ++addrY) {
                    for (int X=0; X<NumInBlocks; ++X) {
                        size_t RowNumPadded = X*InBlockDim + addrX;
                        size_t RowNumLimit  = LayerSize[ID];
                        size_t ColNumPadded = Y*OutBlockDim + addrY;
                        size_t ColNumLimit  = LayerSize[ID+1];
                        if ( (RowNumPadded < RowNumLimit) && (ColNumPadded < ColNumLimit) ) {
                            Observation[RowNum*ObservVecWidth+X] = W[ID][RowNumPadded*OutLayerSize + ColNumPadded];
                        }
                        else Observation[RowNum*ObservVecWidth+X] = 0;
                    }
                    RowNum++;
                }
            }
        }
        // Feed VWeight of Layer[ID]
        for (size_t Y=0; Y<NumOutBlocks; ++Y) {
            for (size_t addrX=0; addrX<InBlockDim; ++addrX) {
                for (size_t addrY=0; addrY<OutBlockDim; ++addrY) {
                    for (size_t X=0; X<NumInBlocks; ++X) {
                        size_t RowNumPadded = X*InBlockDim + addrX;
                        size_t RowNumLimit  = LayerSize[ID];
                        size_t ColNumPadded = Y*OutBlockDim + addrY;
                        size_t ColNumLimit  = LayerSize[ID+1];
                        if ( (RowNumPadded < RowNumLimit) && (ColNumPadded < ColNumLimit) ) {                        
                            Observation[RowNum*ObservVecWidth+X] = VW[ID][RowNumPadded*OutLayerSize + ColNumPadded];
                        }
                        else Observation[RowNum*ObservVecWidth+X] = 0;
                    }
                    RowNum++;
                }
            }
        }
    }
    
    // Feed actual observation data into Observation
    for (size_t iter=0; iter<NumSamples; ++iter) {
        size_t  InBlockDim = BlockDim[0];
        size_t NumInBlocks = NumBlocks[0];
        for (int addrX=0; addrX<InBlockDim; ++addrX) {
            for (int X=0; X<NumInBlocks; ++X) {
                size_t RowNumPadded = X*InBlockDim + addrX;
                size_t RowNumLimit  = LayerSize[0];
                if (RowNumPadded<RowNumLimit) Observation[RowNum*ObservVecWidth+X] = Observ[iter*ObservSpaceDim+RowNumPadded];
                else Observation[RowNum*ObservVecWidth+X] = 0;
            }
            RowNum++;
        }
    }

    // Length of BiasStd Vector
    size_t BiasStdVecLength = PaddedLayerSize[NumLayers-1];
    for (size_t i=1; i<NumLayers; ++i) {
        BiasStdVecLength += 2*PaddedLayerSize[i];
    }
    double * BiasStd = (double *) calloc(BiasStdVecLength, sizeof(double));
    
    // Feed Bias and VBias into BiasStd
    RowNum = 0;
    for (size_t ID=0; ID<NumLayers-1; ++ID) {
        size_t nextLayerDim = PaddedLayerSize[ID+1];
        size_t nextLayerDimLimit = LayerSize[ID+1];
        for (size_t k=0; k<nextLayerDim; ++k) {
            if (k<nextLayerDimLimit) BiasStd[RowNum] = B[ID][k];
            else BiasStd[RowNum] = 0;
            RowNum++;
        }
        for (size_t k=0; k<nextLayerDim; ++k) {
            if (k<nextLayerDimLimit) BiasStd[RowNum] = VB[ID][k];
            else BiasStd[RowNum] = 0;
            RowNum++;
        }
    }
    
    // Feed (1/Std)^2 into BiasStd
    for (size_t k=0; k<PaddedLayerSize[NumLayers-1]; ++k) {
        size_t LayerDimLimit = LayerSize[NumLayers-1];
        if (k<LayerDimLimit) BiasStd[RowNum] = 1.0 / Std[k] / Std[k];
        else BiasStd[RowNum] = 0;
        RowNum++;
    }


    //////////////////// FPGA - Init ////////////////////

    TRPO_WriteDRAM_actions_t write_action;
    write_action.param_start_bytes = 0;
    write_action.param_size_bytes = PaddedObservVecItems * sizeof(double);
    write_action.instream_fromCPU = Observation;
    TRPO_WriteDRAM_run(engine, &write_action);
    fprintf(stderr, "[INFO] Loading Model and Simulation Data...Done\n");


    //////////////////// FPGA - Run ////////////////////
    
    // Here we assume 4 layers
    
    // Number of Cycles to Run - Forward Propagation and Back Propagation
    size_t MaxBlkDim0Dim2     = (BlockDim[0]>BlockDim[2]) ? BlockDim[0] : BlockDim[2];
    size_t FwdCyclesPerSample = BlockDim[0] + (BlockDim[1]-1)*MaxBlkDim0Dim2 + BlockDim[2]*BlockDim[3];
    size_t BwdCyclesPerSample = BlockDim[1]*MaxBlkDim0Dim2 + BlockDim[2]*BlockDim[3];
    size_t CyclesPerSample    = (FwdCyclesPerSample>BwdCyclesPerSample) ? FwdCyclesPerSample : BwdCyclesPerSample;
    size_t PropCyclesTotal    = CyclesPerSample * (NumSamples + 1);
    
    // Number of Cycles to Run - Read Result
    size_t FVPLength = 0;
    for (size_t i=0; i<NumLayers-1; ++i) {
        FVPLength += PaddedLayerSize[i] * PaddedLayerSize[i+1];
        FVPLength += PaddedLayerSize[i+1];
    }
    int PaddedFVPLength = ((int)ceil((double)FVPLength/2))*2;
    
    // Number of Cycles to Run - Total
    size_t NumTicks = WeightInitVecLength + PropCyclesTotal + PaddedFVPLength + 20;

    // Allocation Memory Space for FVP Result
    double * FVPResult = (double *) calloc(PaddedFVPLength, sizeof(double));

    // Init Advanced Static Interface
    TRPO_Run_actions_t run_action;
    run_action.param_NumSamples           = NumSamples;
    run_action.param_PaddedObservVecItems = PaddedObservVecItems;
    run_action.instream_BiasStd           = BiasStd;
    run_action.outstream_FVP              = FVPResult;

    // Run DFE and Measure Elapsed Time
    fprintf(stderr, "[INFO] Running on FPGA for %zu cycles...\n", NumTicks);
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    TRPO_Run_run(engine, &run_action);
    gettimeofday(&tv2, NULL);
    double runtimeS = ((tv2.tv_sec-tv1.tv_sec) * (double)1E6 + (tv2.tv_usec-tv1.tv_usec)) / (double)1E6;
    fprintf(stderr, "[INFO] Running on FPGA...Done\n");
    fprintf(stderr, "[INFO] Elasped Time (FPGA) is %f seconds.\n", runtimeS);
    
    // Free Engine and Maxfile
    max_unload(engine);
    TRPO_free();

    // Read FVP into Result
    pos = 0;
    size_t FVPPos = 0;
    for (size_t i=0; i<NumLayers-1; ++i) {
        size_t  curLayerSizePadded = PaddedLayerSize[i];
        size_t nextLayerSizePadded = PaddedLayerSize[i+1];
        size_t  curLayerSizeReal   = LayerSize[i];
        size_t nextLayerSizeReal   = LayerSize[i+1];
        for (size_t j=0; j<curLayerSizePadded; ++j) {
            for (size_t k=0; k<nextLayerSizePadded; ++k) {
                if ( (j<curLayerSizeReal) && (k<nextLayerSizeReal) ) {
                    Result[pos] = FVPResult[FVPPos];
                    pos++;
                }
                FVPPos++;
            }
        }
        for (size_t k=0; k<nextLayerSizePadded; ++k) {
            if (k<nextLayerSizeReal) {
                Result[pos] = FVPResult[FVPPos];
                pos++;
            }
            FVPPos++;
        }
    }
    for (size_t k=0; k<ActionSpaceDim; ++k) {
        Result[pos] = 2 * NumSamples * VLogStd[k];
        pos++;
    }    

    // Averaging Fisher Vector Product over the samples and apply CG Damping
    for (size_t i=0; i<pos; ++i) {
        Result[i] = Result[i] / (double)NumSamples;
        Result[i] += CG_Damping * Input[i];
    }


    //////////////////// Clean Up ////////////////////

    fprintf(stderr, "[INFO] Clean up...\n");

    // Free Memories Allocated for Reading Files
    for (size_t i=0; i<NumLayers-1; ++i) {
        free(W[i]); free(VW[i]);
        free(B[i]); free(VB[i]);
    }
    free(Observ); free(Mean); free(Std); free(Action); free(Advantage); free(VLogStd);

    // Free Memories Allocated for DFE
    free(Observation); free(BiasStd); free(FVPResult);

    return runtimeS;
}


