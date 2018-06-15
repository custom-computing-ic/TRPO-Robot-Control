#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "omp.h"

#include "TRPO.h"
#include "mujoco.h"
#include "mjmodel.h"
#include "lbfgs.h"


double RunTraining (TRPOparam param, const int NumIter, const size_t NumThreads) {

    //////////////////// Read Parameters ////////////////////

    // OpenMP Settings
    omp_set_num_threads(NumThreads);

    // Assign Parameters
    const size_t NumLayers  = param.NumLayers;
    char * AcFunc           = param.AcFunc;
    size_t * LayerSize      = param.LayerSize;
    const size_t NumSamples = param.NumSamples;
    char * ModelFile        = param.ModelFile;
    char * BaselineFile     = param.BaselineFile;
    const double CG_Damping = param.CG_Damping;
    double ResidualTh       = 1e-10;
    size_t MaxIter          = 10;
    double MaxKL            = 0.01;
    double MaxBackTracks    = 10;
    double AcceptRatio      = 0.1;
    double gamma            = 0.995;
    double lam              = 0.98;

    // Layer Size of Baseline
    size_t LayerSizeBase[4] = {16, 16, 16, 1};


    // Dimension of Observation Space and Action Space
    const size_t ObservSpaceDim = LayerSize[0];
    const size_t ActionSpaceDim = LayerSize[NumLayers-1];

    // Number of Policy Parameters
    size_t NumParams     = NumParamsCalc(param.LayerSize, param.NumLayers);
    size_t NumParamsBase = NumParamsCalc(LayerSizeBase, NumLayers);
    int PaddedParamsBase = (int)ceil((double)NumParamsBase/16.0)*16;

    // iterator when traversing through input vector and result vector
    size_t pos;

    // Number of Episodes per Batch
    const int NumEpBatch = 16;
    
    // Length of Each Episode - timestep_limit
    const int EpLen = 150;
    
    // Length of Each TimeStep (s)
    const double TimeStepLen = 0.02;
     
    // number of generalized coordinates = dim(qpos)
    const int nq = 6;
    
    // Random Seed
    srand(0);

    // pi - for Gaussian Random Number generation
    const double pi = 3.1415926535897931;


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
    double * Reward    = (double *) calloc(NumSamples, sizeof(double));
    double * Return    = (double *) calloc(NumSamples, sizeof(double));
    double * Baseline  = (double *) calloc(NumSamples, sizeof(double));
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


    //////////////////// Memory Allocation - Baseline ////////////////////
    
    // WBase[i]: Weight Matrix from Layer[i] to Layer[i+1]
    // BBase[i]: Bias Vector from Layer[i] to Layer[i+1]
    // Item (j,k) in WBase[i] refers to the weight from Neuron #j in Layer[i] to Neuron #k in Layer[i+1]
    // Item BBase[k] is the bias of Neuron #k in Layer[i+1]
    double * WBase [NumLayers-1];
    double * BBase [NumLayers-1];
    for (size_t i=0; i<NumLayers-1; ++i) {
        WBase[i] = (double *) calloc(LayerSizeBase[i]*LayerSizeBase[i+1], sizeof(double));
        BBase[i] = (double *) calloc(LayerSizeBase[i+1], sizeof(double));
    }

    // GW[i]: Gradient of Loss Function w.r.t to Neural Network Weight W[i]
    // GB[i]: Gradient of Loss Function w.r.t to Neural Network Bias B[i]
    // There is one-to-one correspondence between: GW[i] and W[i], GB[i] and B[i], GStd[i] and Std[i]
    double * GWBase [NumLayers-1];
    double * GBBase [NumLayers-1];
    for (size_t i=0; i<NumLayers-1; ++i) {
        GWBase[i] = (double *) calloc(LayerSizeBase[i]*LayerSizeBase[i+1], sizeof(double));
        GBBase[i] = (double *) calloc(LayerSizeBase[i+1], sizeof(double));
    }
    
    // Layer[i] : Memory of each layer's outputs, i.e. y_i
    // GLayer[i]: Gradient of Loss Function w.r.t. the pre-activation values in Layer[i], i.e. d(Loss)/d(x_i)s
    double * LayerBase  [NumLayers];
    double * GLayerBase [NumLayers];
    for (size_t i=0; i<NumLayers; ++i) {
        LayerBase[i]  = (double *) calloc(LayerSizeBase[i], sizeof(double));
        GLayerBase[i] = (double *) calloc(LayerSizeBase[i], sizeof(double));
    }
    
    // Pamameters Vector for L-BFGS Optimisation
    lbfgsfloatval_t * LBFGS_x = lbfgs_malloc(PaddedParamsBase);
    
    // Objection Function Value
    lbfgsfloatval_t LBFGS_fx;    

    //////////////////// Memory Allocation - MuJoCo Simulation ////////////////////

    // Observation Vector and it Mean and Variance (for Filter)
    double * ob     = (double *) calloc(ObservSpaceDim, sizeof(double));
    double * obMean = (double *) calloc(ObservSpaceDim, sizeof(double));
    double * obVar  = (double *) calloc(ObservSpaceDim, sizeof(double));
    
    // Action Vector (for MuJoCo)
    double * ac     = (double *) calloc(ActionSpaceDim, sizeof(double));


    //////////////////// Parameters for Baseline ////////////////////
    
    TRPOBaselineParam BaselineParam;

    // Paramaters
    BaselineParam.NumLayers      = NumLayers;
    BaselineParam.ObservSpaceDim = ObservSpaceDim;
    BaselineParam.NumEpBatch     = NumEpBatch;
    BaselineParam.EpLen          = EpLen;
    BaselineParam.NumSamples     = NumSamples;
    BaselineParam.NumParams      = NumParamsBase;
    BaselineParam.PaddedParams   = PaddedParamsBase;
    BaselineParam.LayerSizeBase  = LayerSizeBase;
    BaselineParam.AcFunc         = AcFunc;
    
    // For Forward Propagation and Back Propagation
    BaselineParam.WBase          = WBase;
    BaselineParam.BBase          = BBase;
    BaselineParam.LayerBase      = LayerBase;
    BaselineParam.GWBase         = GWBase;
    BaselineParam.GBBase         = GBBase;
    BaselineParam.GLayerBase     = GLayerBase;
    
    // Training Data
    BaselineParam.Observ         = Observ;
    BaselineParam.Target         = Return;    // The prediction target
    BaselineParam.Predict        = Baseline;  // The prediction



    //////////////////// Initialisation - Neural Network ////////////////////
    
    // Note: Here we just initialise the Neural Network from a Datafile
    //       which is the initialisation given by the Python ML Libraries.
    //       We can also initialise the Neural Network ourselves
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


    //////////////////// Initialisation - Baseline ////////////////////
    
    // Note: Here we just initialise the Neural Network from a Datafile
    //       which is the initialisation given by the Python ML Libraries.
    //       We can also initialise the Neural Network ourselves
    // Open Model File that contains Weights, Bias and std
    FILE *BaselineFilePointer = fopen(BaselineFile, "r");
    if (BaselineFilePointer==NULL) {
        fprintf(stderr, "[ERROR] Cannot open BaselineFile [%s]. \n", BaselineFile);
        return -1;
    }
    
    // Read Weights and Bias from file
    for (size_t i=0; i<NumLayers-1; ++i) {
        // Reading Weights W[i]: from Layer[i] to Layer[i+1]
        size_t curLayerDim  = LayerSizeBase[i];
        size_t nextLayerDim = LayerSizeBase[i+1];
        for (size_t j=0; j<curLayerDim;++j) {
            for (size_t k=0; k<nextLayerDim; ++k) {
                fscanf(BaselineFilePointer, "%lf", &WBase[i][j*nextLayerDim+k]);
            }
        }
        // Reading Bias B[i]: from Layer[i] to Layer[i+1]
        for (size_t k=0; k<nextLayerDim; ++k) {
            fscanf(BaselineFilePointer, "%lf", &BBase[i][k]);
        }
    }

    // Close Baseline Model File
    fclose(BaselineFilePointer);


    //////////////////// Initialisation - Advantage ////////////////////

    // Init Z-Filter count
    int obFilterCount = 0;
    int rwFilterCount = 0;

    // L-BFGS Optimisation for Baseline Fitting
    lbfgs_parameter_t LBFGS_Param;
    lbfgs_parameter_init(&LBFGS_Param);
    LBFGS_Param.max_iterations = 25;


    ///////// Init MuJoCo /////////

    // activate software
    mj_activate("mjkey.txt");
    
    // Load and Compile MuJoCo Simulation Model
    char errstr[100] = "[ERROR] Could not load binary model";
    mjModel* m = mj_loadXML("armDOF_0.xml", 0, errstr, 100);
    if(!m) mju_error_s("[ERROR] Load model error: %s", errstr);    

    // Observation Space: Get ID of "DOF1" "DOF2" "wrist" "grip" "object"
    const int DOF1   = mj_name2id(m, mjOBJ_BODY, "DOF1");
    const int DOF2   = mj_name2id(m, mjOBJ_BODY, "DOF2");
    const int wrist  = mj_name2id(m, mjOBJ_BODY, "wrist");
    const int grip   = mj_name2id(m, mjOBJ_BODY, "grip");
    const int object = mj_name2id(m, mjOBJ_BODY, "object");
    
    printf("[INFO] ID: DOF1 DOF2 wrist grip object = %d %d %d %d %d\n", DOF1, DOF2, wrist, grip, object);
    
    // Action Space: Get ID of "M0" "M1" "M2"
    const int M0 = mj_name2id(m, mjOBJ_ACTUATOR, "M0");
    const int M1 = mj_name2id(m, mjOBJ_ACTUATOR, "M1");
    const int M2 = mj_name2id(m, mjOBJ_ACTUATOR, "M2");
    
    printf("[INFO] ID: M0 M1 M2 = %d %d %d\n", M0, M1, M2);


    //////////////////// Main Loop ////////////////////

    // Tic
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);

    // Run Training for NumIter Iterations
    for (int iter=0; iter<NumIter; ++iter) {
    
        // Run Training for several Episodes in each Iteration - can be parallelised
        for (int ep=0; ep<NumEpBatch; ++ep) {

            ///////// Reset Environment /////////
        
            // Init MuJoCo Data - TODO Instead of Make, call reset?
            mjData* d = mj_makeData(m);
            
            // Generate Random Position for the target
            double target_x = ((double)rand()/(double)RAND_MAX) * 0.076 + 0.084;
            double target_y = ((double)rand()/(double)RAND_MAX) * 0.100 - 0.005;
            double target_z = ((double)rand()/(double)RAND_MAX) * 0.100;
            
            d->qpos[nq-3] = target_x;
            d->qpos[nq-2] = target_y;
            d->qpos[nq-1] = target_z;
            
            
            ///////// Rollout Several Time Steps in Each Episode /////////
            
            for(int timeStep=0; timeStep<EpLen; ++timeStep) {
            
                // Row Address of the Simulation Data
                int RowAddr = ep * EpLen + timeStep;
                
                // Get Raw Observation Vector: DOF1 DOF2 wrist grip object
                int ob_pos = 0;
                for (int i=DOF1;   i<DOF1+3;   ++i) ob[ob_pos++] = d->xpos[i];
                for (int i=DOF2;   i<DOF2+3;   ++i) ob[ob_pos++] = d->xpos[i];
                for (int i=wrist;  i<wrist+3;  ++i) ob[ob_pos++] = d->xpos[i];
                for (int i=grip;   i<grip+3;   ++i) ob[ob_pos++] = d->xpos[i];
                for (int i=object; i<object+3; ++i) ob[ob_pos++] = d->xpos[i];
                
                // Filter Observation Space: Z-Filter
                // Update Filter
                obFilterCount++;
                if (obFilterCount == 1) {
                    for (int i=0; i<ObservSpaceDim; ++i) {
                        obMean[i] = ob[i];
                        obVar[i]  = 0;
                    }
                }
                else {
                    for (int i=0; i<ObservSpaceDim; ++i) {
                        double temp = obMean[i];
                        obMean[i] = obMean[i] + (ob[i] - obMean[i]) / (double)obFilterCount;
                        obVar[i]  = obVar[i]  + (ob[i] - obMean[i]) * (ob[i] - temp);
                    }
             
                }
                // Apply Filter
                for (int i=0; i<ObservSpaceDim; ++i) {
                    if (obFilterCount == 1) ob[i] = 0;
                    else {
                        ob[i] = (ob[i]-obMean[i]) / ( sqrt(obVar[i]/(obFilterCount-1)) + 1e-8 );
                        ob[i] = (ob[i] > 5) ? 5 : ( (ob[i]<-5) ? -5 : ob[i] );
                    }
                }

                // Save Data to Observation Matrix
                for (int i=0; i<ObservSpaceDim; ++i) Observ[RowAddr*ObservSpaceDim+i] = ob[i];

                
                ///////// Forward Propagation /////////
                
                // Assign Input Values
                for (size_t i=0; i<ObservSpaceDim; ++i) Layer[0][i] = ob[i];
    
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
                
                // Save Data to Mean Matrix
                for (int i=0; i<ActionSpaceDim; ++i) Mean[RowAddr*ActionSpaceDim+i] = Layer[NumLayers-1][i];
                
                // Save Data to Std Vector - This is actually Redundant
                for (int i=0; i<ActionSpaceDim; ++i) Std[i] = exp(LogStd[i]);
                
                // Get Action from Mean - Sample from the distribution
                for (int i=0; i<ActionSpaceDim; ++i) {
                    // Box-Muller
                    double u1 = rand() * (1.0 / RAND_MAX);
                    double u2 = rand() * (1.0 / RAND_MAX);
                    double z0 = sqrt(-2.0 * log(u1)) * cos(2*pi*u2);
                    //double z1 = sqrt(-2.0 * log(u1)) * sin(2*pi*u2);
                    
                    // N(mu, sigma^2) = N(0,1) * sigma + mu
                    ac[i] = z0 * exp(LogStd[i]) + Layer[NumLayers-1][i];   
                }
                
                // Save Data to Action Matrix
                for (int i=0; i<ActionSpaceDim; ++i) Action[RowAddr*ActionSpaceDim+i] = ac[i];

                
                ///////// Physical Simulation /////////
                
                // Send action to mjData
                d->ctrl[M0] = ac[0];
                d->ctrl[M1] = ac[1];
                d->ctrl[M2] = ac[2];                
                
                // Run MuJoCo Simulation
                mjtNum simStart = d->time;
                while (d->time-simStart < TimeStepLen) mj_step(m, d);

                
                ///////// Calculate Reward /////////
          
                // Get Position of the Grip and the Object
                double   gripPos[3] = {d->xpos[grip], d->xpos[grip+1], d->xpos[grip+2]};
                double objectPos[3] = {d->xpos[object], d->xpos[object+1], d->xpos[object+2]};
                
                // Reward Function  TODO Modify Reward Function with -Eculidean Distance - Action Norm?
                double re = 0;
                for (int i=0; i<3; ++i) {
                    re -= 100 * (objectPos[i] - gripPos[i]) * (objectPos[i] - gripPos[i]);
                    re -= ac[i] * ac[i];
                }
                
                // Save Reward to Reward Vector
                Reward[RowAddr] = re;                            
            
            } // End of Episode

            // Free MuJoCo Data - TODO mj_resetData?
            mj_deleteData(d);
        
        }  // End of the MuJoCo Simulation
        
        ///////// Calculate Mean Episode Rewards  /////////
        
        double EpRewMean = 0;
        for (int i=0; i<NumSamples; ++i) EpRewMean += Reward[i];
        EpRewMean = EpRewMean / (double) NumEpBatch;
        
        printf("[INFO] Iteration %d, Mean Episode Rewards = %f\n", iter, EpRewMean);
        
        
        ///////// Calculate Advantage /////////
        
        // Discount Reward to get Return
        for (int ep=0; ep<NumEpBatch; ++ep) {
        
            // Calculate Return
            for (int currentStep=0; currentStep<EpLen; ++currentStep) {
                // Reward in the current step
                pos = ep*EpLen + currentStep;
                double thisStepReturn = Reward[pos];
                // Discounted future Reward
                for (int futureStep=currentStep+1; futureStep<EpLen; ++futureStep) {
                    thisStepReturn += Reward[ep*EpLen+futureStep] * pow(gamma, futureStep-currentStep);
                }
                Return[pos] = thisStepReturn;
            }
            
            // Using Value Function to estimate return
            
            // For each step in each episode
            for (int currentStep=0; currentStep<EpLen; ++currentStep) {
            
                // Obsevation Vector and Normalised Time Step
                pos = ep*EpLen + currentStep;
                for (int i=0; i<ObservSpaceDim; ++i) {
                    LayerBase[0][i] = Observ[pos*ObservSpaceDim+i];
                }
                LayerBase[0][ObservSpaceDim] = (double) currentStep / (double) EpLen;
                
                // Forward Propagation
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
                            // sigmoid Activation Function
                            case 's': {LayerBase[i+1][j] = 1.0/(1+exp(-LayerBase[i+1][j])); break;}
                            // Default: Activation Function not supported
                            default: {
                                printf("[ERROR] Activation Function for Layer [%zu] is %c. Unsupported.\n", i+1, AcFunc[i+1]);
                                return -1;
                            }
                        }
                    }
                }
                
                // Write result to Baseline
                Baseline[pos] = LayerBase[NumLayers-1][0];
            
            }
            
            // Using Reward to temporarily hold 'deltas'
            for (int currentStep=0; currentStep<EpLen-1; ++currentStep) {
                Reward[ep*EpLen+currentStep] += gamma * Baseline[ep*EpLen+currentStep+1] - Baseline[ep*EpLen+currentStep];
            }
            Reward[ep*EpLen+EpLen-1] += (-1) * Baseline[ep*EpLen+EpLen-1];
            
            // Calculate the Advantage of this episode
            for (int currentStep=0; currentStep<EpLen; ++currentStep) {
                pos = ep*EpLen + currentStep;
                double thisStepAdvantage = Reward[pos];
                for (int futureStep=currentStep+1; futureStep<EpLen; ++futureStep) {
                    thisStepAdvantage += Reward[ep*EpLen+futureStep] * pow(gamma*lam, futureStep-currentStep);
                }
                Advantage[pos] = thisStepAdvantage;
            }            

        }
        
        // Standarise Advantage
        double AdvantageMean = 0;
        for (int i=0; i<NumSamples; ++i) AdvantageMean += Advantage[i];
        AdvantageMean = AdvantageMean / (double) NumSamples;
        
        double AdvantageStd = 0;
        for (int i=0; i<NumSamples; ++i) AdvantageStd += (Advantage[i] - AdvantageMean)*(Advantage[i] - AdvantageMean);
        AdvantageStd = sqrt(AdvantageStd / (double) (NumSamples));
        
        for (int i=0; i<NumSamples; ++i) Advantage[i] = (Advantage[i] - AdvantageMean) / AdvantageStd;
        
        
        ///////// Baseline Update /////////  TODO: Can be executed concurrently with TRPO Update

        // Write weights to LBFGS_x
        pos = 0;
        for (size_t i=0; i<NumLayers-1; ++i) {
            size_t curLayerDim = LayerSizeBase[i];
            size_t nextLayerDim = LayerSizeBase[i+1];
            for (size_t j=0; j<curLayerDim;++j) {
                for (size_t k=0; k<nextLayerDim; ++k) {
                    LBFGS_x[pos] = WBase[i][j*nextLayerDim+k];
                    pos++;
                }
            }
            for (size_t k=0; k<nextLayerDim; ++k) {
                LBFGS_x[pos] = BBase[i][k];
                pos++;
            }
        }      
        
        int LBFGSStatus = lbfgs(BaselineParam.PaddedParams, LBFGS_x, &LBFGS_fx, evaluate, progress, &BaselineParam, &LBFGS_Param);

  
        //////////////////// TRPO Update ////////////////////
  
        ///////// Computing Policy Gradient /////////

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
    
        // Update Model from theta
        pos = 0;
        for (size_t i=0; i<NumLayers-1; ++i) {
            size_t curLayerDim  = LayerSize[i];
            size_t nextLayerDim = LayerSize[i+1];
            for (size_t j=0; j<curLayerDim;++j) {
                for (size_t k=0; k<nextLayerDim; ++k) {
                    W[i][j*nextLayerDim+k] = theta[pos];
                    pos++;
                }
            }
            for (size_t k=0; k<nextLayerDim; ++k) {
                B[i][k] = theta[pos];
                pos++;
            }
        }
        for (size_t k=0; k<ActionSpaceDim; ++k) {
            LogStd[k] = theta[pos];
            pos++;
        }

    
    }

    // Toc
    gettimeofday(&tv2, NULL);
    double runtimeS = ((tv2.tv_sec-tv1.tv_sec) * (double)1E6 + (tv2.tv_usec-tv1.tv_usec)) / (double)1E6;


    //////////////////// Clean Up ////////////////////
    
    // Clean-Up MuJoCo
    mj_deleteModel(m);
    mj_deactivate();
    
    // Clean-Up L-BFGS
    lbfgs_free(LBFGS_x);

    // Model: Weight & Bias, Gradient of Weight & Bias, Policy Gradient of Weight & Bias, R{} Gradient of Weight & Bias
    for (size_t i=0; i<NumLayers-1; ++i) {
        free(W[i]); free(B[i]); 
        free(GW[i]); free(GB[i]);
        free(PGW[i]); free(PGB[i]);
        free(RGW[i]); free(RGB[i]); 
    }
    
    // Model: LogStd, Gradient of LogStd, Policy Gradient of LogStd, R{} Gradient of LogStd
    free(LogStd); free(GLogStd); free(PGLogStd); free(RGLogStd);
    
    // Baseline: Weight & Bias, Gradient of Weight & Bias
    for (size_t i=0; i<NumLayers-1; ++i) {
        free(WBase[i]); free(BBase[i]);
        free(GWBase[i]); free(GBBase[i]);
    }
    
    // Forward and Backward Propagation
    for (size_t i=0; i<NumLayers; ++i) {
        // Model: Ordinary Forward and Backward Propagation
        free(Layer[i]); free(GLayer[i]);
        // Model: Pearlmutter Forward and Backward Propagation
        free(RxLayer[i]); free(RyLayer[i]); free(RGLayer[i]);
        // Baseline: Ordinary Forward and Backward Propagation
        free(LayerBase[i]); free(GLayerBase[i]);
    }

    // Conjugate Gradient
    free(b); free(p); free(r); free(x); free(z);

    // Line Search
    free(fullstep); free(xnew); free(theta);

    // MuJoCo: Observation, Action and Observation Filtering
    free(ob); free(ac); free(obMean); free(obVar);

    // Simulation Data and Advantage Calculation
    free(Observ); free(Mean); free(Std); free(Action); free(Reward); free(Return); free(Baseline); free(Advantage);
    
    return runtimeS;
}





