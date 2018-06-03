#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "omp.h"

#include "TRPO.h"


double CG(TRPOparam param, double *Result, double *b, size_t MaxIter, double ResidualTh, size_t NumThreads){

    //////////////////// Conjugate Gradient ////////////////////

    // This function implements Conjugate Gradient algorithm to solve linear equation Ax=b
    //     Result: The Conjugate Gradient Result, i.e. solution x to Ax=b
    //          b: Vector b in the equation Ax=b
    //    MaxIter: Maximum Iterations of Conjugate Gradient (in modular_rl is 10)
    // ResidualTh: Threshold of Residual (in modular_rl is 1e-10)

    // OpenMP Settings
    omp_set_num_threads(NumThreads);

    // Memory Allocation
    size_t NumParams = NumParamsCalc(param.LayerSize, param.NumLayers);
    double * p = (double *) calloc(NumParams, sizeof(double));
    double * r = (double *) calloc(NumParams, sizeof(double));
    double * x = (double *) calloc(NumParams, sizeof(double));
    double * z = (double *) calloc(NumParams, sizeof(double));

    // Initialisation
    double rdotr = 0;
    for (size_t i=0; i<NumParams; ++i) {
        p[i] = b[i];
        r[i] = b[i];
        rdotr += r[i] * r[i];
    }
    
    // Iterative Solver

    // Measure Elapsed Time
    struct timeval tv1, tv2;
    double ComptimeS = 0;
    
    for (size_t iter=0; iter<=MaxIter; ++iter) {

        // Calculate Frobenius Norm of x
        double FrobNorm = 0;
        gettimeofday(&tv1, NULL);
        #pragma omp parallel for reduction (+:FrobNorm)
        for (size_t i=0; i<NumParams; ++i) {
            FrobNorm += x[i] * x[i];
        }
        FrobNorm = sqrt(FrobNorm);
        gettimeofday(&tv2, NULL);
        printf("CG Iter[%zu] Residual Norm=%.12e, Soln Norm=%.12e\n", iter, rdotr, FrobNorm);
        
        // Check Termination Condition
        if (rdotr<ResidualTh || iter==MaxIter) {
            for (size_t i=0; i<NumParams; ++i) Result[i] = x[i];
            break;
        }
        
        // Calculate z = FIM*p
        double FVPTime = FVPFast(param, z, p, NumThreads);
        if (FVPTime<0) {
            fprintf(stderr, "[ERROR] Fisher Vector Product Calculation Failed.\n");
            free(p); free(r); free(x); free(z);
            return -1;
        }
        else {
            ComptimeS += ((tv2.tv_sec-tv1.tv_sec) * (double)1E6 + (tv2.tv_usec-tv1.tv_usec)) / (double)1E6; 
            ComptimeS += FVPTime;
        }
        
        // Update x and r
        double pdotz = 0;
        gettimeofday(&tv1, NULL);
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
        
        gettimeofday(&tv2, NULL);
        ComptimeS += ((tv2.tv_sec-tv1.tv_sec) * (double)1E6 + (tv2.tv_usec-tv1.tv_usec)) / (double)1E6; 
    }
    
    // Clean Up
    free(p); free(r); free(x); free(z);
    
    return ComptimeS;
}

