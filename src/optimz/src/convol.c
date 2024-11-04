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
                        tmpPxlVal += _inputImageMat[tmpPos][j] * _kernel[k + kHalfSize];
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
                        tmpPxlVal += _inputImageMat[tmpPos][j] * _kernel[k + kHalfSize];

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


/*// Horizontal step of separable convolution using AVX/NEON
void convolve_horizontal_avx(const uint8_t* input, uint8_t* output, uint8_t* kH, uint8_t w, uint8_t h, uint8_t kSize) {
    int c = kSize/2;
    for (int i = 0;i<h;i++) {
        for (int j = 0;j<w;j++) {
            int sum = 0;
            for(int k = -c; k <= c;k++) {
                int jj = j + k;
                if (jj >= 0 && jj < w) {
                    sum += input[i * w + jj] * kH[k + c];
                }
            }
            output[i * w + j] = sum;
        }
    }
    uint8_t pad = kSize / 2;

    for (int32_t i = 0; i < h; i++) {
        for (int32_t j = 0; j < w; j += 32) {
            __m256i result = _mm256_setzero_si256();  // Initialize AVX register to 0

            for (uint8_t k = -pad; k <= pad; k++) {
                // Load the input pixels
                __m256i pixel = _mm256_loadu_epi8(&input[y * w + (x + k)]);
                // Set the kernel value for multiplication
                __m256i kernel_val = _mm256_set1_epi8(kH[k + pad]);
                // Accumulate the results
                result = _mm256_add_epi8(result, _mm256_mul_epi32(pixel, kernel_val));
            }

            // Store the result back to the output image
            _mm256_storeu_ps(&output[y * w + x], result);  // Store result back
        }
    }
}

// Vertical pass of separable convolution using AVX/NEON
void convolve_vertical_avx(uint8_t* input, uint8_t* output, uint8_t* kV, uint8_t w, uint8_t h, uint8_t kSize) {
    int c = kSize/2;
    for (int i = 0;i<h;i++) {
        for (int j = 0;j<w;j++) {
            int sum = 0;
            for(int k = -c; k <= c;k++) {
                int jj = j + k;
                if (jj >= 0 && jj < w) {
                    sum += input[i * w + jj] * kV[k + c];
                }
            }
            output[i * w + j] = sum;
        }
    }
    uint8_t pad = kSize / 2;

    for (uint32_t y = pad; y < height - pad; y++) {
        for (uint32_t x = 0; x < width; x += 8) {
            __m256 result = _mm256_setzero_ps();  // Initialize AVX register to 0

            // Apply vertical kernel
            for (int k = -pad; k <= pad; k++) {
                // Load the input pixels
                __m256 pixel = _mm256_loadu_ps(&input[(y + k) * width + x]);
                // Set the kernel value for multiplication
                __m256 kernel_val = _mm256_set1_ps(kV[k + pad]);
                // Accumulate the results
                result = _mm256_add_ps(result, _mm256_mul_ps(pixel, kernel_val));
            }

            // Store the result back to the output image
            _mm256_storeu_ps(&output[y * width + x], result);  // Store result back
        }
    }
}

// Main function for the 2D convolution using AVX/NEON
void separable_cnvol_AVX(int* input, int* output, double* kernel, int w, int h, int kSize, int __pFlag) {
    
    // Allocate memory for intermediate buffer
    uint8_t* _intBuf = (uint8_t*)malloc(w * h * sizeof(uint8_t));
    if (_intBuf == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Split the 2D kernel into 2x1D kernels
    uint8_t* _verKernel = (uint8_t*)malloc( kSize * sizeof(uint8_t));
    uint8_t* _horizKernel = (uint8_t*)malloc( kSize * sizeof(uint8_t));
    _verKernel[0] = 1;
    _verKernel[1] = 2;
    _verKernel[2] = 1;

    _horizKernel[0] = 1;
    _horizKernel[1] = 2;
    _horizKernel[2] = 1;


    // Apply the convolution on both axis
    convolve_horizontal_avx(input, _intBuf, _horizKernel, w, h,3);
    convolve_vertical_avx(_intBuf, output, _verKernel, w, h, 3);

    // Free intermediate buffer
    if(_intBuf) {
        free(_intBuf);
    }

    // Free vertical kernel
    if(_verKernel) {
        free(_verKernel);
    }

    // Free horizontal kernel
    if(_horizKernel) {
        free(_horizKernel);
    }
}
*/

// Separable convolution w/out AVX/NEON
int** separable_convol(int** inputImage, double* hK, double* vK, int w, int h, int kSize, int _pFlag) {
    double **_horizConvoRes = (double**)malloc(h * sizeof(double*));
    for (int i = 0; i < h; i++) {
        _horizConvoRes[i] = (double*)malloc(w * sizeof(double));
    }

    int **_finalConvoRes = (int**)malloc(h * sizeof(int*));
    for (int i = 0; i < h; i++) {
        _finalConvoRes[i] = (int*)malloc(w * sizeof(int));
    }

    // Apply convolution on each axis
    basic_horizConvol(inputImage, _horizConvoRes, hK, w, h, kSize, _pFlag);
    basic_vertConvol(_horizConvoRes, _finalConvoRes, vK, w, h, kSize, _pFlag);

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