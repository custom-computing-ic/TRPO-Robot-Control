#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "omp.h"

#include "TRPO.h"

/*
#include "Maxfiles.h"
#include "MaxSLiCInterface.h"

void Test_FVP(size_t NumThreads)
{
	
    // ArmDOF_0-v0
    char AcFunc [] = {'l', 't', 't', 'l'};
    size_t LayerSize [] = {15, 16, 16, 3};

    char * ModelFileName = "ArmTestModel.txt";
    char * DataFileName  = "ArmTestData.txt";
    char * FVPFileName   = "ArmTestFVP.txt";

    TRPOparam Param;
    Param.ModelFile  = ModelFileName;
    Param.DataFile   = DataFileName;
    Param.NumLayers  = 4;
    Param.AcFunc     = AcFunc;
    Param.LayerSize  = LayerSize;
    Param.NumSamples = 2400;
    Param.CG_Damping = 0.1;

    // Open Simulation Data File that contains test data
    FILE *DataFilePointer = fopen(FVPFileName, "r");
    if (DataFilePointer==NULL) {
        fprintf(stderr, "[ERROR] Cannot open Data File [%s]. \n", FVPFileName);
        return;
    }

    // Memory Allocation
    size_t NumParams = NumParamsCalc(Param.LayerSize, Param.NumLayers);
    double * input   = (double *) calloc(NumParams, sizeof(double));
    double * result  = (double *) calloc(NumParams, sizeof(double));
    double * expect  = (double *) calloc(NumParams, sizeof(double)); 
    
    // Read Input and Expect
    for (size_t i=0; i<NumParams; ++i) {
         fscanf(DataFilePointer, "%lf %lf", &input[i], &expect[i]);
    }
    fclose(DataFilePointer);

    double FVPStatus = FVPFast(Param, result, input, NumThreads);
    if (FVPStatus<0) fprintf(stderr, "[ERROR] Fisher Vector Product Calculation Failed.\n");
    
    // Check Result
    double percentage_err = 0;
    for (size_t i=0; i<NumParams; ++i) {        
        double cur_err = fabs( (result[i]-expect[i])/expect[i] ) * 100;
    	if (expect[i] != 0) percentage_err += cur_err;
    	if (cur_err>1) printf("FVP[%zu]=%e, Expect=%e. %.4f%% Difference\n", i, result[i], expect[i], cur_err);
    }
    percentage_err = percentage_err / (double)NumParams;
    printf("--------------------- Swimmer Test (%zu Threads) ----------------------\n", NumThreads);
    printf("[INFO] Fisher Vector Product Mean Absolute Percentage Error = %.12f%%\n", percentage_err);
    printf("---------------------------------------------------------------------\n\n");

    // Clean Up    
    free(input); free(result); free(expect);
    
    return;
}


void Test_CG(size_t NumThreads)
{
	
    // ArmDOF_0-v0
    char AcFunc [] = {'l', 't', 't', 'l'};
    size_t LayerSize [] = {15, 16, 16, 3};

    char * ModelFileName = "ArmTestModel.txt";
    char * DataFileName  = "ArmTestData.txt";
    char * CGFileName    = "ArmTestCG.txt";

    TRPOparam Param;
    Param.ModelFile  = ModelFileName;
    Param.DataFile   = DataFileName;
    Param.NumLayers  = 4;
    Param.AcFunc     = AcFunc;
    Param.LayerSize  = LayerSize;
    Param.NumSamples = 2400;
    Param.CG_Damping = 0.1;

    // Open Simulation Data File that contains test data
    FILE *DataFilePointer = fopen(CGFileName, "r");
    if (DataFilePointer==NULL) {
        fprintf(stderr, "[ERROR] Cannot open Data File [%s]. \n", CGFileName);
        return;
    }

    // Memory Allocation
    size_t NumParams = NumParamsCalc(Param.LayerSize, Param.NumLayers);
    double * input   = (double *) calloc(NumParams, sizeof(double));
    double * result  = (double *) calloc(NumParams, sizeof(double));
    double * expect  = (double *) calloc(NumParams, sizeof(double)); 
    
    // Read Input and Expect
    for (size_t i=0; i<NumParams; ++i) {
         fscanf(DataFilePointer, "%lf %lf", &input[i], &expect[i]);
    }
    fclose(DataFilePointer);
    
    printf("----------------------- ArmDOF CG Test (%zu Threads) ------------------------\n", NumThreads);
    double compTime = CG(Param, result, input, 10, 1e-10, NumThreads);
    if (compTime<0) fprintf(stderr, "[ERROR] Conjugate Gradient Calculation Failed.\n");
    
    // Check Result
    double percentage_err = 0;
    for (size_t i=0; i<NumParams; ++i) {        
        double cur_err = fabs( (result[i]-expect[i])/expect[i] ) * 100;
    	if (expect[i] != 0) percentage_err += cur_err;
    	if (cur_err>1) printf("CG[%zu]=%e, Expect=%e. %.4f%% Difference\n", i, result[i], expect[i], cur_err);
    }
    percentage_err = percentage_err / (double)NumParams;
    printf("\n[INFO] CPU Computing Time = %f seconds\n", compTime);
    printf("[INFO] Conjugate Gradient Mean Absolute Percentage Error = %.4f%%\n", percentage_err);
    printf("---------------------------------------------------------------------\n\n");

    // Clean Up    
    free(input); free(result); free(expect);
    
    return;
}


void Test_FVP_FPGA() {


    // ArmDOF_0-v0
    char            AcFunc [] = {'l', 't', 't', 'l'};
    size_t       LayerSize [] = { 15, 16, 16, 3};
    size_t PaddedLayerSize [] = { 16, 16, 16, 4};
    size_t       NumBlocks [] = {  4,  4,  4, 2};

    char * ModelFileName = "ArmTestModel.txt";
    char * DataFileName  = "ArmTestData.txt";
    char * FVPFileName   = "ArmTestFVP.txt";

    TRPOparam Param;
    Param.ModelFile         = ModelFileName;
    Param.DataFile          = DataFileName;
    Param.NumLayers         = 4;
    Param.AcFunc            = AcFunc;
    Param.LayerSize         = LayerSize;
    Param.PaddedLayerSize   = PaddedLayerSize;
    Param.NumBlocks         = NumBlocks;
    Param.NumSamples        = 2400;
    Param.CG_Damping        = 0.1;

    // Open Simulation Data File that contains test data
    FILE *DataFilePointer = fopen(FVPFileName, "r");
    if (DataFilePointer==NULL) {
        fprintf(stderr, "[ERROR] Cannot open Data File [%s]. \n", FVPFileName);
        return;
    }

    // Memory Allocation
    size_t NumParams = NumParamsCalc(Param.LayerSize, Param.NumLayers);
    double *       input = (double *) calloc(NumParams, sizeof(double));
    double *  CPU_output = (double *) calloc(NumParams, sizeof(double));
    double * FPGA_output = (double *) calloc(NumParams, sizeof(double));
    
    // Read Input
    for (size_t i=0; i<NumParams; ++i) {
        double temp;
        fscanf(DataFilePointer, "%lf %lf", &input[i], &temp);
    }
    fclose(DataFilePointer);

    //////////////////// CPU ////////////////////

    int FVPStatus = FVP(Param, CPU_output, input);
    if (FVPStatus!=0) fprintf(stderr, "[ERROR] Fisher Vector Product Calculation Failed.\n");

    //////////////////// FPGA ////////////////////

    double runtimeFPGA = FVP_FPGA(Param, FPGA_output, input);

    //////////////////// Check Results ////////////////////
  
    // Check Result
    double percentage_err = 0;
    for (size_t i=0; i<NumParams; ++i) {        
        double cur_err = fabs( (FPGA_output[i]-CPU_output[i])/CPU_output[i] ) * 100;
    	if (CPU_output[i] != 0) percentage_err += cur_err;
    	if (cur_err>1) {
    	    printf("FVP_FPGA[%zu]=%e, FVP_CPU[%zu]=%e. %.12f%% Difference\n", i, FPGA_output[i], i, CPU_output[i], cur_err);
    	}
    }
    
    // Print Results
    FILE *ResultFilePointer = fopen("result.txt", "w");
    if(ResultFilePointer == NULL) fprintf(stderr, "[ERROR] Open Output File Failed.\n");
    for (size_t i=0; i<NumParams; ++i) {
        fprintf(ResultFilePointer, "CPU_output[%4zu] = % 014.12f, FPGA_output[%4zu] = % 014.12f\n", i, CPU_output[i], i, FPGA_output[i]);
    }
    fclose(ResultFilePointer);
    
    percentage_err = percentage_err / (double)NumParams;
    printf("--------------------------- Test FPGA ---------------------------\n");
    printf("[INFO] FPGA Computing Time = %f seconds\n", runtimeFPGA);
    printf("[INFO] Mean Absolute Percentage Error = %.12f%%\n", percentage_err);
    printf("---------------------------------------------------------------------\n\n");


    // Clean Up    
    free(input); free(CPU_output); free(FPGA_output);
    
    return;

}

void Test_CG_FPGA(size_t NumThreads)
{


    // ArmDOF_0-v0
    char            AcFunc [] = {'l', 't', 't', 'l'};
    size_t       LayerSize [] = { 15, 16, 16, 3};
    size_t PaddedLayerSize [] = { 16, 16, 16, 4};
    size_t       NumBlocks [] = {  4,  4,  4, 2};

    char * ModelFileName = "ArmTestModel.txt";
    char * DataFileName  = "ArmTestData.txt";
    char * CGFileName    = "ArmTestCG.txt";


    TRPOparam Param;
    Param.ModelFile         = ModelFileName;
    Param.DataFile          = DataFileName;
    Param.NumLayers         = 4;
    Param.AcFunc            = AcFunc;
    Param.LayerSize         = LayerSize;
    Param.PaddedLayerSize   = PaddedLayerSize;
    Param.NumBlocks         = NumBlocks;
    Param.NumSamples        = 2400;
    Param.CG_Damping        = 0.1;

    // Open Simulation Data File that contains test data
    FILE *DataFilePointer = fopen(CGFileName, "r");
    if (DataFilePointer==NULL) {
        fprintf(stderr, "[ERROR] Cannot open Data File [%s]. \n", CGFileName);
        return;
    }

    // Memory Allocation
    size_t NumParams     = NumParamsCalc(Param.LayerSize, Param.NumLayers);
    double * input       = (double *) calloc(NumParams, sizeof(double));
    double * CPU_output  = (double *) calloc(NumParams, sizeof(double));
    double * FPGA_output = (double *) calloc(NumParams, sizeof(double)); 
    
    // Read Input and Expect
    double placeholder;
    for (size_t i=0; i<NumParams; ++i) {
         fscanf(DataFilePointer, "%lf %lf", &input[i], &placeholder);
    }
    fclose(DataFilePointer);

    // FPGA-based CG Calculation    
    printf("\n---------------------- CG Test FPGA (%zu Threads) -----------------------\n", NumThreads);
    double runtimeFPGA = CG_FPGA(Param, FPGA_output, input, 10, 1e-10, NumThreads);
    if (runtimeFPGA<0) fprintf(stderr, "[ERROR] FPGA-based Conjugate Gradient Calculation Failed.\n");

    // CPU-based CG Calculation
    printf("---------------------- CG Test CPU (%zu Threads) -----------------------\n", NumThreads);
    double runtimeCPU = CG(Param, CPU_output, input, 10, 1e-10, NumThreads);
    if (runtimeCPU<0) fprintf(stderr, "[ERROR] CPU-based Conjugate Gradient Calculation Failed.\n");
    
    // Check Result
    double percentage_err = 0;
    double max_percentage_err = 0;
    for (size_t i=0; i<NumParams; ++i) {        
        double cur_err = fabs( (FPGA_output[i]-CPU_output[i])/CPU_output[i] ) * 100.0;
    	if (CPU_output[i] != 0) {
    	    percentage_err += cur_err;
    	    max_percentage_err = (max_percentage_err > cur_err) ? max_percentage_err : cur_err;
    	}
//    	if (cur_err>1) printf("CG_FPGA[%zu]=%e, CG_CPU[%zu]=%e. %.4f%% Difference\n", i, FPGA_output[i], i, CPU_output[i], cur_err);
    }
    
    // Print Results
    FILE *ResultFilePointer = fopen("result.txt", "w");
    if(ResultFilePointer == NULL) fprintf(stderr, "[ERROR] Open Output File Failed.\n");
    for (size_t i=0; i<NumParams; ++i) {
        fprintf(ResultFilePointer, "%.12f %.12f\n", CPU_output[i], FPGA_output[i]);
    }
    fclose(ResultFilePointer);    
    
    percentage_err = percentage_err / (double)NumParams;
    printf("\n-------------------------- CG Result Check --------------------------\n");
    printf("[INFO] FPGA Time = %f seconds, CPU Time = %f seconds\n", runtimeFPGA, runtimeCPU);
    printf("[INFO] Mean Absolute Percentage Error = %.12f%%, Max Percentage Error = %.12f%%\n", percentage_err, max_percentage_err);
    printf("---------------------------------------------------------------------\n\n");

    // Clean Up    
    free(input); free(CPU_output); free(FPGA_output);
    
    return;
}


void Test_TRPO_Update(size_t NumThreads)
{
	
    // ArmDOF_0-v0
    char AcFunc [] = {'l', 't', 't', 'l'};
    size_t LayerSize [] = {15, 16, 16, 3};

    char * ModelFileName  = "ArmTestModel.txt";
    char * DataFileName   = "ArmTestData.txt";
    char * ResultFileName = "ArmTestModelUpdated.txt";

    TRPOparam Param;
    Param.ModelFile  = ModelFileName;
    Param.DataFile   = DataFileName;
    Param.NumLayers  = 4;
    Param.AcFunc     = AcFunc;
    Param.LayerSize  = LayerSize;
    Param.NumSamples = 2400;
    Param.CG_Damping = 0.1;

    // Open Data File that contains updated Model Parameters
    FILE *DataFilePointer = fopen(ResultFileName, "r");
    if (DataFilePointer==NULL) {
        fprintf(stderr, "[ERROR] Cannot open Data File [%s]. \n", ResultFileName);
        return;
    }

    // Memory Allocation
    size_t NumParams = NumParamsCalc(Param.LayerSize, Param.NumLayers);
    double * result  = (double *) calloc(NumParams, sizeof(double));
    double * expect  = (double *) calloc(NumParams, sizeof(double)); 
    
    // Read Input and Expect
    for (size_t i=0; i<NumParams; ++i) {
         fscanf(DataFilePointer, "%lf", &expect[i]);
    }
    fclose(DataFilePointer);
    
    printf("----------------------- TRPO Update Test (%zu Threads) ------------------------\n", NumThreads);
    double compTime = TRPO_Update(Param, result, NumThreads);
    if (compTime<0) fprintf(stderr, "[ERROR] TRPO Update Failed.\n");

    // Check Result
    double percentage_err = 0;
    for (size_t i=0; i<NumParams; ++i) {        
        double cur_err = fabs( (result[i]-expect[i])/expect[i] ) * 100;
    	if (expect[i] != 0) percentage_err += cur_err;
    	if (cur_err>1) printf("Actual[%zu]=%e, Expect=%e. %.4f%% Difference\n", i, result[i], expect[i], cur_err);
    }
    percentage_err = percentage_err / (double)NumParams;
    printf("\n[INFO] CPU Computing Time = %f seconds\n", compTime);
    printf("[INFO] TRPO Update Mean Absolute Percentage Error = %.4f%%\n", percentage_err);
    printf("---------------------------------------------------------------------\n\n");

    // Clean Up    
    free(result); free(expect);
    
    return;
}
*/
/*
void Test_TRPO_MuJoCo(const int NumIterations, const size_t NumThreads){
	
    // ArmDOF_0-v0
    char AcFunc []          = {'l', 't', 't', 'l'};
    size_t LayerSize []     = {15, 16, 16, 3};
    char * ModelFileName    = "ArmTestModel.txt";
    char * BaselineFileName = "ArmTestBaseline.txt";
    char * ResultFileName   = "ArmTrainingResult";

    TRPOparam Param;
    Param.ModelFile    = ModelFileName;
    Param.BaselineFile = BaselineFileName;
    Param.ResultFile   = ResultFileName;
    Param.NumLayers    = 4;
    Param.AcFunc       = AcFunc;
    Param.LayerSize    = LayerSize;
    Param.NumSamples   = 2400;
    Param.CG_Damping   = 0.1;

    
    printf("----------------------- Run TRPO Training (%zu Threads) ------------------------\n", NumThreads);
    double compTime = TRPO_MuJoCo (Param, NumIterations, NumThreads);
    if (compTime<0) fprintf(stderr, "[ERROR] TRPO Update Failed.\n");
    printf("\n[INFO] CPU TRPO Training Time = %f seconds\n", compTime);
    printf("---------------------------------------------------------------------\n\n");
    
    return;
}
*/
void Test_TRPO_Lightweight(const int NumIterations, const size_t NumThreads){
	
    // ArmDOF_0-v0
    char AcFunc []          = {'l', 't', 't', 'l'};
    size_t LayerSize []     = {15, 16, 16, 3};
    char * ModelFileName    = "ArmTestModel.txt";
    char * BaselineFileName = "ArmTestBaseline.txt";
    char * ResultFileName   = "ArmTrainingResult";

    TRPOparam Param;
    Param.ModelFile    = ModelFileName;
    Param.BaselineFile = BaselineFileName;
    Param.ResultFile   = ResultFileName;
    Param.NumLayers    = 4;
    Param.AcFunc       = AcFunc;
    Param.LayerSize    = LayerSize;
    Param.NumSamples   = 2400;
    Param.CG_Damping   = 0.1;

    
    printf("----------------------- Run TRPO Training (%zu Threads) ------------------------\n", NumThreads);
    double compTime = TRPO_Lightweight (Param, NumIterations, NumThreads);
    if (compTime<0) fprintf(stderr, "[ERROR] TRPO Update Failed.\n");
    printf("\n[INFO] CPU TRPO Training Time = %f seconds\n", compTime);
    printf("---------------------------------------------------------------------\n\n");
    
    return;
}

int main() {

    // ArmDOF_0-v0
    TRPOparam Param;
    char AcFunc[4]      = {'l', 't', 't', 'l'};
    size_t LayerSize[4] = {15, 16, 16, 3};
    Param.NumLayers     = 4;
    Param.AcFunc        = AcFunc;
    Param.LayerSize     = LayerSize;

    //////////////////// Fisher Vector Product Computation ////////////////////
    
//    Test_FVP(6);
//    Test_CG(6);
//    Test_TRPO_Update(6);

    //////////////////// FPGA ////////////////////

//    Test_FVP_FPGA();
//    Test_CG_FPGA(6);

    //////////////////// Simulation Based Training ////////////////////

    //Test_TRPO_MuJoCo(101, 6);
    Test_TRPO_Lightweight(201, 6);
    //TRPO_Video(Param, "Arm200.txt");

    return 0;
}

