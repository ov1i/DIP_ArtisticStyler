#pragma once
#ifndef _CONVOLS_PACK_H         // _CONVOLS_PACK_H
#define _CONVOLS_PACK_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>

#ifdef __ARM_NEON
    #include <arm_neon.h>  // For ARM based processors (ex. Mac M1, M2...)
    #include <sys/auxv.h>
    #include <asm/hwcap.h>
    void horizConvol_NEON(int **_inputImageMat, double **_outputImageMat, double *_kernel, int w, int h, int kSize, int _pFlag);
    void vertConvol_NEON(double **_inputImageMat, int **_outputImageMat, double *_kernel, int w, int h, int kSize, int _pFlag);
#elif defined(__AVX__)
    #include <immintrin.h>  // For INTEL/AMD based processors (ex. Intel i5 9500U, ...)
    void horizConvol_AVX(int **_inputImageMat, double** _outputImageMat, double *_kernel, int w, int h, int kSize, int _pFlag);
    void vertConvol_AVX(double **_inputImageMat, int **_outputImageMat, double *_kernel, int w, int h, int kSize, int _pFlag);
#endif

// LIB data types
typedef struct
{
   int **_inputImageMat;
   int width;
   int height;
}t_imageDPack;

typedef struct
{
    double **_kernel2D;
    double *_kernelHorizontal;
    double *_kernelVertical;
    int kernelSize;
}t_kernelDPack;

typedef struct
{
    int _paddingFlag;
    int _convoTypeFlag;
}t_flagsDPack;      

// !LIB data types

// LIB FCs
int* v_imagePadAdapter(int* _inputVec, int w, int h, int kSize, int _pFlag);
int** m_imagePadAdapterINT(int** _inputMat, int w, int h, int kSize, int _pFlag);
double** m_imagePadAdapterDOUBLE(double** _inputMat, int w, int h, int kSize, int _pFlag);
void basic_horizConvol(int **_inputImageMat, double** _outputImageMat, double *_kernel, int w, int h, int kSize, int _pFlag);
void basic_vertConvol(double **_inputImageMat, int **_outputImageMat, double *_kernel, int w, int h, int kSize, int _pFlag);
int** separable_convol(int** inputImage, double* hK, double* vK, int w, int h, int kSize, int _pFlag);
int** basic_convol(int **_inputImageMat, double **kernel, int w, int h, int kSize, int _pFlag);
// LIB MAIN FC
int** covolveWrapper(t_imageDPack __imgDataPack, t_kernelDPack __kDataPack, t_flagsDPack __fDPack);
// !LIB MAIN FC
// !LIB FCs

// LIB MOCK FCs
int mockSum(int a, int b);
void mockHelloWorld();
// !LIB MOCK FCs

#endif                      // !_CONVOLS_PACK_H