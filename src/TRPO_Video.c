#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "omp.h"

#include "TRPO.h"
#include "mujoco.h"
#include "glfw3.h"

#include "svpng.inc"


int TRPO_Video (TRPOparam param, char * ModelFile) {

    //////////////////// Read Parameters ////////////////////

    // Assign Parameters
    const size_t NumLayers  = param.NumLayers;
    char * AcFunc           = param.AcFunc;
    size_t * LayerSize      = param.LayerSize;

    // Dimension of Observation Space and Action Space
    const size_t ObservSpaceDim = LayerSize[0];
    const size_t ActionSpaceDim = LayerSize[NumLayers-1];

    // iterator when traversing through input vector and result vector
    size_t pos;
    
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
    double * _W [NumLayers-1];
    double * _B [NumLayers-1];
    for (size_t i=0; i<NumLayers-1; ++i) {
        _W[i] = (double *) calloc(LayerSize[i]*LayerSize[i+1], sizeof(double));
        _B[i] = (double *) calloc(LayerSize[i+1], sizeof(double));
    }
    
    // LogStd[i] is the log of std[i] in the policy
    double * LogStd = (double *) calloc(ActionSpaceDim, sizeof(double));

    // Layer[i] : Memory of each layer's outputs, i.e. y_i
    double * Layer  [NumLayers];
    for (size_t i=0; i<NumLayers; ++i) {
        Layer[i]  = (double *) calloc(LayerSize[i], sizeof(double));
    }


    //////////////////// Memory Allocation - Observation and Action ////////////////////

    double * ob = (double *) calloc(ObservSpaceDim, sizeof(double));
    double * ac = (double *) calloc(ActionSpaceDim, sizeof(double));


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
                fscanf(ModelFilePointer, "%lf", &_W[i][j*nextLayerDim+k]);
            }
        }
        // Reading Bias B[i]: from Layer[i] to Layer[i+1]
        for (size_t k=0; k<nextLayerDim; ++k) {
            fscanf(ModelFilePointer, "%lf", &_B[i][k]);
        }
    }

    // Read LogStd from file
    for (size_t k=0; k<ActionSpaceDim; ++k) {
        fscanf(ModelFilePointer, "%lf", &LogStd[k]);
    }

    // Close Model File
    fclose(ModelFilePointer);


    ///////// Init OpenGL /////////

    // init GLFW
    if( !glfwInit() ) mju_error("Could not initialize GLFW");

    // create invisible window, single-buffered
    glfwWindowHint(GLFW_VISIBLE, 0);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(640, 480, "Invisible window", NULL, NULL);
    if(!window) mju_error("Could not create GLFW window");

    // make context current
    glfwMakeContextCurrent(window);


    ///////// Init MuJoCo /////////

    // activate software
    mj_activate("mjkey.txt");
    
    // Load and Compile MuJoCo Simulation Model
    char errstr[100] = "[ERROR] Could not load binary model";
    mjModel* m = mj_loadXML("armDOF_0.xml", 0, errstr, 100);
    if(!m) mju_error_s("[ERROR] Load model error: %s", errstr);    

    // MuJoCo visualization
    mjvScene scn;
    mjvCamera cam;
    mjvOption opt;
    mjrContext con;

    // initialize MuJoCo visualization
    mjv_makeScene(&scn, 1000);
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjr_defaultContext(&con);
    mjr_makeContext(m, &con, 200);

    // center and scale view
    cam.lookat[0] = m->stat.center[0];
    cam.lookat[1] = m->stat.center[1];
    cam.lookat[2] = m->stat.center[2];
    cam.distance = 1.5 * m->stat.extent;

    // set rendering to offscreen buffer
    mjr_setBuffer(mjFB_OFFSCREEN, &con);
    if( con.currentBuffer!=mjFB_OFFSCREEN )
        printf("Warning: offscreen rendering not supported, using default/window framebuffer\n");

    // get size of active renderbuffer
    mjrRect viewport = mjr_maxViewport(&con);
    int W = viewport.width;
    int H = viewport.height;

    printf("[INFO] Video Resolution is %d*%d\n", W, H);

    // allocate rgb and depth buffers
    unsigned char* rgb = (unsigned char*)malloc(3*W*H);
    float* depth = (float*)malloc(sizeof(float)*W*H);
    if(!rgb || !depth ) mju_error("Could not allocate buffers");

    // create output rgb file
    char ResultFileName[30];
    strcpy(ResultFileName, ModelFile);
    char suffix[8];
    strcpy(suffix, ".out");
    strcat(ResultFileName, suffix);
    FILE* fp = fopen(ResultFileName, "wb");
    if(!fp) mju_error("Could not open rgbfile for writing");

    // Init MuJoCo Data
    mjData* d = mj_makeData(m);
            
    // Generate Random Position for the target
    double target_x = ((double)rand()/(double)RAND_MAX) * 0.076 + 0.084 + 0.01;
    double target_y = ((double)rand()/(double)RAND_MAX) * 0.100 - 0.05;
    double target_z = ((double)rand()/(double)RAND_MAX) * 0.100;
            
    d->qpos[nq-3] = target_x;
    d->qpos[nq-2] = target_y;
    d->qpos[nq-1] = target_z;
            
    // Apply Forward Kinematics to calculate xpos based on qpos
    mj_forward(m, d);

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
    
    for(int timeStep=0; timeStep<EpLen; ++timeStep) {

        ///////// Render New Frame /////////
        
        // update abstract scene
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);

        // render scene in offscreen buffer
        mjr_render(viewport, &scn, &con);

        // add time stamp in upper-left corner
        char stamp[50];
        sprintf(stamp, "Time = %.3f", d->time);
        mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, stamp, NULL, &con);

        // read rgb and depth buffers
        mjr_readPixels(rgb, depth, viewport, &con);

        // write rgb image to file
        fwrite(rgb, 3, W*H, fp);

        // print every 10 frames: '.' if ok, 'x' if OpenGL error
        if(timeStep%10==0) {
            if(mjr_getError()) printf("x");
            else printf(".");
        }

        // Create a PNG Screenshot every 50 frames
        // TODO Image seems to be flipped vertically and mirrored
        /*
        if(timeStep%50==0) {
            char PNGFileName[30];
            strcpy(PNGFileName, ModelFile);
            char PNGsuffix[10];
            sprintf(PNGsuffix, "%d.png", timeStep);
            strcat(PNGFileName, PNGsuffix);        
            FILE *fp1 = fopen(PNGFileName, "wb");
            svpng(fp1, 640, 480, rgb, 0);
            fclose(fp1);
        }
        */

        ///////// Robot Controller /////////

        // Get Raw Observation Vector: DOF1 DOF2 wrist grip object
        int ob_pos = 0;
        for (int i=0; i<3; ++i) ob[ob_pos++] = d->xpos[3*DOF1+i];
        for (int i=0; i<3; ++i) ob[ob_pos++] = d->xpos[3*DOF2+i];
        for (int i=0; i<3; ++i) ob[ob_pos++] = d->xpos[3*wrist+i];
        for (int i=0; i<3; ++i) ob[ob_pos++] = d->xpos[3*grip+i];
        for (int i=0; i<3; ++i) ob[ob_pos++] = d->xpos[3*object+i];
                
        // Forward Propagation
        for (size_t i=0; i<ObservSpaceDim; ++i) Layer[0][i] = ob[i];
        
        for (size_t i=0; i<NumLayers-1; ++i) {
            
            // Propagate from Layer[i] to Layer[i+1]
            for (size_t j=0; j<LayerSize[i+1]; ++j) {
                
                // Calculating pre-activated value for item[j] in next layer
                Layer[i+1][j] = _B[i][j];
                for (size_t k=0; k<LayerSize[i]; ++k) {
                    // From Neuron #k in Layer[i] to Neuron #j in Layer[i+1]
                    Layer[i+1][j] += Layer[i][k] * _W[i][k*LayerSize[i+1]+j];
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
                
        // Get Action from Mean - Sample from the distribution
        for (int i=0; i<ActionSpaceDim; ++i) {
            // Box-Muller
            double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
            double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
            double z0 = sqrt(-2.0 * log(u1)) * cos(2*pi*u2);
            //double z1 = sqrt(-2.0 * log(u1)) * sin(2*pi*u2);
                    
            // N(mu, sigma^2) = N(0,1) * sigma + mu
            ac[i] = z0 * exp(LogStd[i]) + Layer[NumLayers-1][i];
        }


        ///////// Physical Simulation /////////
                
        // Send action to mjData
        d->ctrl[M0] = ac[0];
        d->ctrl[M1] = ac[1];
        d->ctrl[M2] = ac[2];
                
        // Run MuJoCo Simulation
        mj_step(m, d);
    }

    printf("\n[INFO] Run the following command:\n");
    printf("ffmpeg -f rawvideo -pixel_format rgb24 -video_size 640x480 -framerate 50 -i %s -vf \"vflip\" %s.mp4 \n", ResultFileName, ModelFile);


    //////////////////// Clean Up ////////////////////

    // close file, free buffers
    fclose(fp);
    free(rgb);
    free(depth);

    // Clean-Up MuJoCo
    mj_deleteData(d);
    mj_deleteModel(m);
    mjr_freeContext(&con);
    mjv_freeScene(&scn);
    mj_deactivate();

    // Terminate OpenGL
    glfwTerminate();

    // Model: Weight, Bias, LogStd
    for (size_t i=0; i<NumLayers-1; ++i) {
        free(_W[i]); free(_B[i]);
    }
    free(LogStd);

    // Model: Forward Propagation
    for (size_t i=0; i<NumLayers; ++i) {
        free(Layer[i]);
    }

    // MuJoCo: Observation, Action
    free(ob); free(ac);

    return 0;
}





