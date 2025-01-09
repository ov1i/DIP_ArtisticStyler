#include "convol.h"

// Add padding to vector format Image
int* v_imagePadAdapter(int* _inputVec, int w, int h, int kSize, int _pFlag) {
    int padSize = kSize / 2;

    int paddedW = w + 2 * padSize;
    int paddedH = h + 2 * padSize;

    int* _paddedImg = (int*)calloc(paddedW * paddedH, sizeof(int));

    // Add 0 Padding
    if(_pFlag == 0) {
        int tempIdx = padSize + padSize * paddedW;
        for (int i = 0; i < h * w; i+=w) {
            int *dest = &_paddedImg[tempIdx];
            int *src = &_inputVec[i];

            // Copy the input image over the new padded image
            memcpy(dest, src, w * sizeof(int));

            tempIdx += paddedW;
        }
    }
    else {
        if(_pFlag != 1) {
            printf("\nWarning: Invalid flag, continuing with default value(replicate-padding)");
        }

        int tempIdx = padSize + padSize * paddedW;
        
        for (int i = 0; i < h * w; i+=w) {
            int *dest = &_paddedImg[tempIdx];
            int *src = &_inputVec[i];

            // Copy the input image over the new padded image
            memcpy(dest, src, w * sizeof(int));

            tempIdx += paddedW;
        }


        // Copy the special edge cases
        for (int i = 0; i < padSize; i++) {
            memcpy(&_paddedImg[i * paddedW + padSize], &_inputVec[0], w * sizeof(int));
            memcpy(&_paddedImg[(((paddedH * (paddedW - 1)) - (i * paddedW)) + padSize)], &_inputVec[h*(w-1)], w * sizeof(int));
        }

        for (int i = 0; i < padSize; i++) {
            for(int j = 0; j < paddedH; j++) {
                _paddedImg[j * paddedW + i] = _paddedImg[padSize];
                _paddedImg[j * paddedW + (paddedW - padSize + i)] = _paddedImg[j * paddedW + (paddedW - padSize - 1)];
            }
        }
    }

    return _paddedImg;
}

// Add padding to matrix format Image (INTEGER)
int** m_imagePadAdapterINT(int** _inputMat, int w, int h, int kSize, int _pFlag) {
    int padSize = kSize / 2;

    int paddedW = w + 2 * padSize;
    int paddedH = h + 2 * padSize;

    int** _paddedImgMat = (int**)calloc(paddedH, sizeof(int*));
    for(int i = 0; i<paddedH;i++) {
        _paddedImgMat[i] = (int*)calloc(paddedW, sizeof(int));
    }
    
    // Add 0 Padding
    if(_pFlag == 0) {
        for (int i = 0; i < h; i++) { 
            // Copy the input image over the new padded image
            memcpy(&_paddedImgMat[i+padSize][padSize], _inputMat[i], w * sizeof(int));
        }
    }
    else {
        if(_pFlag != 1) {
            printf("\nWarning: Invalid flag, continuing with default value(replicate-padding)");
        }
        for (int i = 0; i < h; i++) {
            // Copy the input image over the new padded image
            memcpy(&_paddedImgMat[i+padSize][padSize], _inputMat[i], w * sizeof(int));
        }


        // Copy the special edge cases
        for (int i = 0; i < padSize; i++) {
            memcpy(&_paddedImgMat[i][padSize], _inputMat[0], w * sizeof(int));
            memcpy(&_paddedImgMat[paddedH-i-1][padSize], _inputMat[h-1], w * sizeof(int));
        }
        for(int i = 0; i < paddedH; i++) {
            for (int j = 0; j < padSize; j++) {
                _paddedImgMat[i][j] = _paddedImgMat[i][padSize];
                _paddedImgMat[i][paddedW - j - 1] = _paddedImgMat[i][paddedW - padSize - 1];
            }
        }
    }

    return _paddedImgMat;
}

// Add padding to matrix format Image (DOUBLE)
double** m_imagePadAdapterDOUBLE(double** _inputMat, int w, int h, int kSize, int _pFlag) {
    int padSize = kSize / 2;

    int paddedW = w + 2 * padSize;
    int paddedH = h + 2 * padSize;

    // Create new image with padding
    double** _paddedImgMat = (double**)calloc(paddedH, sizeof(double*));
    for(int i = 0; i<paddedH;i++) {
        _paddedImgMat[i] = (double*)calloc(paddedW, sizeof(double));
    }
    
    // Add 0 Padding
    if(_pFlag == 0) {
        for (int i = 0; i < h; i++) { 
            // Copy the input image over the new padded image
            memcpy(&_paddedImgMat[i+padSize][padSize], _inputMat[i], w * sizeof(double));
        }
    }
    else {
        if(_pFlag != 1) {
            printf("\nWarning: Invalid flag, continuing with default value(replicate-padding)");
        }
        for (int i = 0; i < h; i++) {
            // Copy the input image over the new padded image
            memcpy(&_paddedImgMat[i+padSize][padSize], _inputMat[i], w * sizeof(double));
        }


        // Copy the special edge cases
        for (int i = 0; i < padSize; i++) {
            memcpy(&_paddedImgMat[i][padSize], _inputMat[0], w * sizeof(double));
            memcpy(&_paddedImgMat[paddedH-i-1][padSize], _inputMat[h-1], w * sizeof(double));
        }
        for(int i = 0; i < paddedH; i++) {
            for (int j = 0; j < padSize; j++) {
                _paddedImgMat[i][j] = _paddedImgMat[i][padSize];
                _paddedImgMat[i][paddedW - j - 1] = _paddedImgMat[i][paddedW - padSize - 1];
            }
        }
    }

    return _paddedImgMat;
}

// Horizontal step of separable convolution w/out AVX/NEON
void basic_horizConvol(int **_inputImageMat, double** _outputImageMat, double *_kernel, int w, int h, int kSize, int _pFlag) {

    int kHalfSize = kSize / 2;

    // Zero-Padding case
    if(_pFlag == 0) {
        for(int i = 0; i < h; i++) {
            for(int j = 0; j < w; j++) {
                double tmpPxlVal = 0.0;
                for(int k = -kHalfSize; k < kHalfSize + 1; k++) {
                    int tmpPos = j + k;
                              
                    if (tmpPos >= 0 && tmpPos < w)
                        tmpPxlVal += _inputImageMat[i][tmpPos] * _kernel[k + kHalfSize];
                }
                _outputImageMat[i][j] = tmpPxlVal;
            }
        }
    }
    // Replicate-Padding case
    else { 
        if(_pFlag != 1) {
            printf("\nWarning: Invalid flag, continuing with default value(replicate-padding)");
        }
        for(int i = 0; i < h; i++) {
            for(int j = 0; j < w; j++) {
                double tmpPxlVal = 0.0;
                for(int k = -kHalfSize; k < kHalfSize + 1; k++) {
                    int tmpPos = j + k;

                    if (tmpPos < 0) tmpPos = 0;
                    else if (tmpPos >= w) tmpPos = w - 1;

                    tmpPxlVal += _inputImageMat[i][tmpPos] * _kernel[k + kHalfSize];
                }
                _outputImageMat[i][j] = tmpPxlVal;
            }
        }
    }
}

// Vertical step of separable convolution w/out AVX/NEON
void basic_vertConvol(double **_inputImageMat, int **_outputImageMat, double *_kernel, int w, int h, int kSize, int _pFlag) {

    int kHalfSize = kSize / 2;
    
    // Zero-Padding case
    if(_pFlag == 0) {
        for(int i = 0; i < h; i++) {
            for(int j = 0; j < w; j++) {
                double tmpPxlVal = 0.0;
                for(int k = -kHalfSize; k < kHalfSize + 1; k++) {
                    int tmpPos = i + k;

                    if (tmpPos >= 0 && tmpPos < h)
                        tmpPxlVal += _inputImageMat[i][tmpPos] * _kernel[k + kHalfSize];

                }
                _outputImageMat[i][j] = (int)round(tmpPxlVal);
            }
        }
    }
    // Replicate-Padding case
    else {
        if(_pFlag != 1) {
            printf("\nWarning: Invalid flag, continuing with default value(replicate-padding)");
        }
        for(int i = 0; i < h; i++) {
            for(int j = 0; j < w; j++) {
                double tmpPxlVal = 0.0;
                for(int k = -kHalfSize; k < kHalfSize + 1; k++) {
                    int tmpPos = i + k;

                    if (tmpPos < 0) tmpPos = 0;
                    else if (tmpPos >= h) tmpPos = h - 1;

                    tmpPxlVal += _inputImageMat[tmpPos][j] * _kernel[k + kHalfSize];
                }
                _outputImageMat[i][j] = (int)round(tmpPxlVal);
            }
        }
    }
}

#ifdef __AVX__
// Horizontal convolution using AVX
void horizConvol_AVX(int **_inputImageMat, double **_outputImageMat, double *_kernel, int w, int h, int kSize, int _pFlag) {
    int kHalfSize = kSize / 2;

    // Parallelize the rows
    #pragma omp parallel for
    for (int i = 0; i < h; i++) { 
        for (int j = 0; j < w; j += 4) { 
            __m256d tmpPxlVal = _mm256_setzero_pd();

            for (int k = -kHalfSize; k <= kHalfSize; k++) {
                int tmpPos = j + k;
                __m256d kernelVal = _mm256_set1_pd(_kernel[k + kHalfSize]);
                __m256d inputVal;

                if (tmpPos >= 0 && tmpPos + 3 < w) {
                    // Aligned memory access for main pixels
                    inputVal = _mm256_set_pd(
                        (double)_inputImageMat[i][tmpPos + 3],
                        (double)_inputImageMat[i][tmpPos + 2],
                        (double)_inputImageMat[i][tmpPos + 1],
                        (double)_inputImageMat[i][tmpPos]
                    );
                } else {
                    // Handle padding using masks
                    double padded[4] = {0.0, 0.0, 0.0, 0.0};
                    for (int idx = 0; idx < 4; idx++) {
                        int pos = tmpPos + idx;
                        if (pos >= 0 && pos < w) {
                            padded[idx] = (double)_inputImageMat[i][pos];
                        }
                    }
                    inputVal = _mm256_loadu_pd(padded);
                }

                tmpPxlVal = _mm256_fmadd_pd(kernelVal, inputVal, tmpPxlVal);
            }

            // Store results back
            _mm256_storeu_pd(&_outputImageMat[i][j], tmpPxlVal);
        }
    }
}

// Vertical convolution using AVX
void vertConvol_AVX(double **_inputImageMat, int **_outputImageMat, double *_kernel, int w, int h, int kSize, int _pFlag) {
    int kHalfSize = kSize / 2;

    #pragma omp parallel for
    for (int j = 0; j < w; j++) {
        for (int i = 0; i < h; i += 4) {
            __m256d tmpPxlVal = _mm256_setzero_pd();

            for (int k = -kHalfSize; k <= kHalfSize; k++) {
                int tmpPos = i + k;
                __m256d kernelVal = _mm256_set1_pd(_kernel[k + kHalfSize]);
                __m256d inputVal;

                if (tmpPos >= 0 && tmpPos + 3 < h) {
                    inputVal = _mm256_set_pd(
                        _inputImageMat[tmpPos + 3][j],
                        _inputImageMat[tmpPos + 2][j],
                        _inputImageMat[tmpPos + 1][j],
                        _inputImageMat[tmpPos][j]
                    );
                } else {
                    // Handle padding
                    double padded[4] = {0.0, 0.0, 0.0, 0.0};
                    for (int idx = 0; idx < 4; idx++) {
                        int pos = tmpPos + idx;
                        if (pos >= 0 && pos < h) {
                            padded[idx] = _inputImageMat[pos][j];
                        }
                    }
                    inputVal = _mm256_loadu_pd(padded);
                }

                tmpPxlVal = _mm256_fmadd_pd(kernelVal, inputVal, tmpPxlVal);
            }

            // Store results back
            double tmpArray[4];
            _mm256_storeu_pd(tmpArray, tmpPxlVal);
            for (int idx = 0; idx < 4; idx++) {
                if (i + idx < h) {
                    _outputImageMat[i + idx][j] = (int)round(tmpArray[idx]);
                }
            }
        }
    }
}

#endif

#ifdef __ARM_NEON
// Horizontal convolution using NEON
void horizConvol_NEON(int **_inputImageMat, double **_outputImageMat, double *_kernel, int w, int h, int kSize) {
    int kHalfSize = kSize / 2;

    // Parallelize rows if desired
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j += 4) {
            float64x2_t tmpPxlVal1 = vdupq_n_f64(0.0); // Initialize Neon registers to 0
            float64x2_t tmpPxlVal2 = vdupq_n_f64(0.0);

            for (int k = -kHalfSize; k <= kHalfSize; k++) {
                int tmpPos = j + k;

                float64x2_t kernelVal = vdupq_n_f64(_kernel[k + kHalfSize]);

                float64x2_t inputVal1, inputVal2;

                if (tmpPos >= 0 && tmpPos + 3 < w) {
                    // Aligned access for the input matrix
                    double temp1[2] = {
                        (double)_inputImageMat[i][tmpPos],
                        (double)_inputImageMat[i][tmpPos + 1]};
                    double temp2[2] = {
                        (double)_inputImageMat[i][tmpPos + 2],
                        (double)_inputImageMat[i][tmpPos + 3]};

                    inputVal1 = vld1q_f64(temp1);
                    inputVal2 = vld1q_f64(temp2);

                } else {
                    // Handle edge cases with zero-padding
                    double padded1[2] = {0.0, 0.0};
                    double padded2[2] = {0.0, 0.0};

                    for (int idx = 0; idx < 2; idx++) {
                        int pos = tmpPos + idx;
                        if (pos >= 0 && pos < w) {
                            padded1[idx] = (double)_inputImageMat[i][pos];
                        }
                    }

                    for (int idx = 2; idx < 4; idx++) {
                        int pos = tmpPos + idx;
                        if (pos >= 0 && pos < w) {
                            padded2[idx - 2] = (double)_inputImageMat[i][pos];
                        }
                    }

                    inputVal1 = vld1q_f64(padded1);
                    inputVal2 = vld1q_f64(padded2);
                }

                tmpPxlVal1 = vfmaq_f64(tmpPxlVal1, kernelVal, inputVal1); // FMA operation
                tmpPxlVal2 = vfmaq_f64(tmpPxlVal2, kernelVal, inputVal2); // FMA operation
            }

            // Store the result back to the output matrix
            double result1[2], result2[2];
            vst1q_f64(result1, tmpPxlVal1);
            vst1q_f64(result2, tmpPxlVal2);

            _outputImageMat[i][j] = result1[0];
            if (j + 1 < w)
                _outputImageMat[i][j + 1] = result1[1];
            if (j + 2 < w)
                _outputImageMat[i][j + 2] = result2[0];
            if (j + 3 < w)
                _outputImageMat[i][j + 3] = result2[1];
        }
    }
}

// Vertical convolution using NEON
void vertConvol_NEON(double **_inputImageMat, int **_outputImageMat, double *_kernel, int w, int h, int kSize) {
    int kHalfSize = kSize / 2;

    for (int j = 0; j < w; j++) {
        for (int i = 0; i < h; i += 4) {
            float64x2_t tmpPxlVal1 = vdupq_n_f64(0.0);
            float64x2_t tmpPxlVal2 = vdupq_n_f64(0.0);

            for (int k = -kHalfSize; k <= kHalfSize; k++) {
                int tmpPos = i + k;

                float64x2_t kernelVal = vdupq_n_f64(_kernel[k + kHalfSize]);

                float64x2_t inputVal1, inputVal2;

                if (tmpPos >= 0 && tmpPos + 3 < h) {
                    double temp1[2] = {
                        _inputImageMat[tmpPos][j],
                        _inputImageMat[tmpPos + 1][j]};
                    double temp2[2] = {
                        _inputImageMat[tmpPos + 2][j],
                        _inputImageMat[tmpPos + 3][j]};

                    inputVal1 = vld1q_f64(temp1);
                    inputVal2 = vld1q_f64(temp2);
                } else {
                    // Handle edge cases with zero-padding
                    double padded1[2] = {0.0, 0.0};
                    double padded2[2] = {0.0, 0.0};

                    for (int idx = 0; idx < 2; idx++) {
                        int pos = tmpPos + idx;
                        if (pos >= 0 && pos < h) {
                            padded1[idx] = _inputImageMat[pos][j];
                        }
                    }

                    for (int idx = 2; idx < 4; idx++) {
                        int pos = tmpPos + idx;
                        if (pos >= 0 && pos < h) {
                            padded2[idx - 2] = _inputImageMat[pos][j];
                        }
                    }

                    inputVal1 = vld1q_f64(padded1);
                    inputVal2 = vld1q_f64(padded2);
                }

                tmpPxlVal1 = vfmaq_f64(tmpPxlVal1, kernelVal, inputVal1);
                tmpPxlVal2 = vfmaq_f64(tmpPxlVal2, kernelVal, inputVal2);
            }

            // Store the result back to the output matrix
            double result1[2], result2[2];
            vst1q_f64(result1, tmpPxlVal1);
            vst1q_f64(result2, tmpPxlVal2);

            if (i < h)
                _outputImageMat[i][j] = (int)round(result1[0]);
            if (i + 1 < h)
                _outputImageMat[i + 1][j] = (int)round(result1[1]);
            if (i + 2 < h)
                _outputImageMat[i + 2][j] = (int)round(result2[0]);
            if (i + 3 < h)
                _outputImageMat[i + 3][j] = (int)round(result2[1]);
        }
    }
}
#endif

// Separable convolution main
int** separable_convol(int** inputImage, double* hK, double* vK, int w, int h, int kSize, int _pFlag) {
    double **_horizConvoRes = (double**)malloc(h * sizeof(double*));
    for (int i = 0; i < h; i++) {
        _horizConvoRes[i] = (double*)malloc(w * sizeof(double));
    }

    int **_finalConvoRes = (int**)malloc(h * sizeof(int*));
    for (int i = 0; i < h; i++) {
        _finalConvoRes[i] = (int*)malloc(w * sizeof(int));
    }

    int useSIMD = 0;

    #ifdef __AVX__
    if (__builtin_cpu_supports("avx")) {
        useSIMD = 1;  // AVX is supported
    }
    #elif defined(__ARM_NEON)
    char buffer[256];
    size_t buffer_size = sizeof(buffer);

    if (sysctlbyname("hw.optional.neon", &buffer, &buffer_size, NULL, 0) == 0) {
        useSIMD = 1; // NEON is supported
    }

    #endif

    // Apply convolution based on availability of SIMD
    if (useSIMD) {
        // Use AVX/NEON-optimized methods
        #ifdef __AVX__
            horizConvol_AVX(inputImage, _horizConvoRes, hK, w, h, kSize, _pFlag);
            vertConvol_AVX(_horizConvoRes, _finalConvoRes, vK, w, h, kSize, _pFlag);
        #elif defined(__ARM_NEON)
            horizConvol_NEON(inputImage, _horizConvoRes, hK, w, h, kSize);
            vertConvol_NEON(_horizConvoRes, _finalConvoRes, vK, w, h, kSize);
        #endif
    } else {
        // Fall back to basic method if no SIMD support is available
        basic_horizConvol(inputImage, _horizConvoRes, hK, w, h, kSize, _pFlag);
        basic_vertConvol(_horizConvoRes, _finalConvoRes, vK, w, h, kSize, _pFlag);
    }

    // Clear interm. buffer (WILL have performance impact/boost on large amount of input data)
    for (int i = 0; i < h; i++) {
        free(_horizConvoRes[i]);
    }
    free(_horizConvoRes);
    
    return _finalConvoRes;
}

// Classic/Basic convolution
int** basic_convol(int **_inputImageMat, double **kernel, int w, int h, int kSize, int _pFlag) {
    int **_outputImageMat = (int**)malloc(h * sizeof(int*));
    for (int i = 0; i < h; i++) {
        _outputImageMat[i] = (int*)malloc(w * sizeof(int));      
    }

    int** _paddedImg = m_imagePadAdapterINT(_inputImageMat, w, h, kSize, _pFlag);
    int kHalfSize = kSize / 2;
        
    for(int i = 0; i < h; i++) {
        for(int j = 0; j < w; j++) {
            double res = 0.0;
            for(int x = -kHalfSize; x < kHalfSize + 1; x++) {
                for(int y = -kHalfSize; y < kHalfSize + 1; y++) {
                    res += _paddedImg[i + x + kHalfSize][j + y + kHalfSize] * kernel[x + kHalfSize][y + kHalfSize];
                }
            }         
            _outputImageMat[i][j] = (int)res;
        }
    }

    return _outputImageMat;
}


// LIB MAIN

// Convolution Wrapper
int** covolveWrapper(t_imageDPack __imgDPack, t_kernelDPack __kDPack, t_flagsDPack __fDPack) {
    if(__fDPack._convoTypeFlag == 1) {
        printf("\nINFO: Convolution method ->> SEPARATED <<-- (param value (%d))\n\n", __fDPack._convoTypeFlag);
        printf("\nINFO: Used KERNEL shape ->> 1D <<- \n\n");
        return separable_convol(__imgDPack._inputImageMat, __kDPack._kernelHorizontal, __kDPack._kernelVertical, __imgDPack.width, __imgDPack.height, __kDPack.kernelSize, __fDPack._paddingFlag);
    }
    else {
        if(__fDPack._convoTypeFlag != 0) {
            printf("\nWARNING: Invalid \"_convoTypeFlag\" choosen value (%d)..\nContinuing with default value\n\n", __fDPack._convoTypeFlag);
            __fDPack._convoTypeFlag = 0;
        }
        printf("\nINFO: Convolution method ->> CLASSIC <<-- (param value (%d) = default)\n\n", __fDPack._convoTypeFlag);
        printf("\nINFO: Used KERNEL shape ->> 2D <<- \n\n");
        return basic_convol(__imgDPack._inputImageMat, __kDPack._kernel2D, __imgDPack.width, __imgDPack.height, __kDPack.kernelSize, __fDPack._paddingFlag);
    }
    return NULL;
}

 // Test function for python linkage
 int mockSum(int a, int b) {
    return a + b;
 }
 // Test function for python linkage
  void mockHelloWorld() {
    printf("Hello world!");
 }

// !LIB MAIN