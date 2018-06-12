#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "omp.h"

#include "TRPO.h"

#include "mujoco.h"
#include "mjmodel.h"


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
    char * DataFile         = param.DataFile;
    const double CG_Damping = param.CG_Damping;
    double ResidualTh       = 1e-10;
    size_t MaxIter          = 10;
    double MaxKL            = 0.01;
    double MaxBackTracks    = 10;
    double AcceptRatio      = 0.1;
    double gamma            = 0.995;

    // Layer Size of Baseline
    size_t LayerSizeBase[4] = {16, 16, 16, 1};


    // Dimension of Observation Space and Action Space
    const size_t ObservSpaceDim = LayerSize[0];
    const size_t ActionSpaceDim = LayerSize[NumLayers-1];

    // Number of Policy Parameters
    size_t NumParams = NumParamsCalc(param.LayerSize, param.NumLayers);

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
    
    // Layer[i] : Memory of each layer's outputs, i.e. y_i
    double * LayerBase [NumLayers];
    for (size_t i=0; i<NumLayers; ++i) {
        LayerBase[i] = (double *) calloc(LayerSizeBase[i], sizeof(double));
    }    



    //////////////////// Memory Allocation - MuJoCo Simulation ////////////////////

    // Observation Vector and it Mean and Variance (for Filter)
    double * ob     = (double *) calloc(ObservSpaceDim, sizeof(double));
    double * obMean = (double *) calloc(ObservSpaceDim, sizeof(double));
    double * obVar  = (double *) calloc(ObservSpaceDim, sizeof(double));
    
    // Action Vector (for MuJoCo)
    double * ac     = (double *) calloc(ActionSpaceDim, sizeof(double));


    //////////////////// Initialisation ////////////////////
    
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

    // Init Z-Filter count
    int obFilterCount = 0;
    int rwFilterCount = 0;


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
    
    // Action Space: Get ID of "j0" "j1" "j2"
    const int j0 = mj_name2id(m, mjOBJ_ACTUATOR, "j0");
    const int j1 = mj_name2id(m, mjOBJ_ACTUATOR, "j1");
    const int j2 = mj_name2id(m, mjOBJ_ACTUATOR, "j2");
    
    printf("[INFO] ID: J0 J1 J2 = %d %d %d\n", j0, j1, j2);


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
            
            // Check qpos and qvel
            for (int i=0; i<m->nq; ++i) printf("%f ", d->qpos[i]);
            printf("\n");
            for (int i=0; i<m->nv; ++i) printf("%f ", d->qvel[i]);
            printf("\n");
            
            
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
                    double ac[i] = z0 * exp(LogStd[i]) + Layer[NumLayers-1][i];   
                }
                
                // Save Data to Action Matrix
                for (int i=0; i<ActionSpaceDim; ++i) Action[RowAddr*ActionSpaceDim+i] = ac[i];

                
                ///////// Physical Simulation /////////
                
                // Send action to mjData
                d->ctrl[j0] = ac[0];
                d->ctrl[j1] = ac[1];
                d->ctrl[j2] = ac[2];                
                
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
        
            // For each step in each episode
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
            
            // Using Value Function to calculate return
            
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
            Reward[ep*EpLen+EpLen-1] += (gamma-1) * Baseline[ep*EpLen+EpLen-1];
            
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
        AdvantageStd = AdvantageStd / (double) (NumSamples-1);
        
        for (int i=0; i<NumSamples; ++i) Advantage[i] = (Advantage[i] - AdvantageMean) / AdvantageStd;
        
        
        
        
        
        
        
        ///////// TRPO Update /////////
        
    
    }

    // Toc
    gettimeofday(&tv2, NULL);
    double runtimeS = ((tv2.tv_sec-tv1.tv_sec) * (double)1E6 + (tv2.tv_usec-tv1.tv_usec)) / (double)1E6;

    // free MuJoCo model, deactivate
    mj_deleteModel(m);
    mj_deactivate();

    // free memory
    free(ob); free(obMean); free(obVar); free(ac);
    
    //
    free(Reward); free(Return); free(Baseline);
    
    // TODO Free WBase[i] BBase[i]
    
    return runtimeS;
}





