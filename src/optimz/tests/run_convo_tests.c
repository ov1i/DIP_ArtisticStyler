#include <time.h>
#include <stdio.h>
#include "../src/convol.h"

// Main function for testing
int main() {
    int w = 512, h = 512;
    int k_S = 5; 
    int pFlag = 1;
    int convoTypeFlag = 1;

    // Allocate memory for mock input data
    int **input = (int**)malloc(h * sizeof(int*));
    for (int i = 0; i < h; i++) {
        input[i] = (int*)malloc(w * sizeof(int));
        for(int j = 0; j < w; j++) {
            input[i][j] = 255;  // Fill with some test data
        }
    }
    double **k = (double **)malloc(k_S * sizeof(double *));
    for (int i = 0; i < k_S; i++) {
        k[i] = (double *)malloc(k_S * sizeof(double));
    }
    double *k_horiz = (double *)malloc(k_S * sizeof(double));
    double *k_vert = (double *)malloc(k_S * sizeof(double));
    // !Allocate memory for mock input data

    if (input == NULL || k == NULL || k_horiz == NULL || k_vert == NULL) {
        printf("Memory allocation failed\n");
        return -1;
    }




    double k_values[5][5] = {
        {0.0029690167439504977, 0.013306209891013656, 0.02193823127971465, 0.013306209891013656, 0.0029690167439504977},
        {0.013306209891013656, 0.05963429543618016,  0.0983203313488458,  0.05963429543618016,  0.013306209891013656},
        {0.02193823127971465,  0.0983203313488458,   0.1621028216371267,  0.0983203313488458,   0.02193823127971465},
        {0.013306209891013656, 0.05963429543618016,  0.0983203313488458,  0.05963429543618016,  0.013306209891013656},
        {0.0029690167439504977, 0.013306209891013656, 0.02193823127971465, 0.013306209891013656, 0.0029690167439504977}
    };
    double k_values_horiz[5] = {0.05448868, 0.24420134, 0.40261995, 0.24420134, 0.05448868};
    double k_values_vert[5] = {0.05448868, 0.24420134, 0.40261995, 0.24420134, 0.05448868};

    // Set data for prev. allocated memory of mock input data
    for (int i = 0; i < k_S; i++) {
        for (int j = 0; j < k_S; j++) {
            k[i][j] = k_values[i][j];   
        }
    }
    
    for (int i = 0; i < k_S; i++) {
        k_horiz[i] = k_values_horiz[i];
        k_vert[i] = k_values_vert[i];
    }
    // !Set data for prev. allocated memory of mock input data


    t_imageDPack mock_ImgDPack;
    t_kernelDPack mock_KerDPack;
    t_flagsDPack mock_FlagsDPack;

    // PACK data into dedicated data types
    mock_ImgDPack._inputImageMat = input;
    mock_ImgDPack.width = w;
    mock_ImgDPack.height = h;
    mock_KerDPack._kernel2D = k;
    mock_KerDPack._kernelHorizontal = k_horiz;
    mock_KerDPack._kernelVertical = k_vert;
    mock_KerDPack.kernelSize = k_S;
    mock_FlagsDPack._paddingFlag = pFlag;
    mock_FlagsDPack._convoTypeFlag = convoTypeFlag;
    // !PACK data into dedicated data types




    // printf("\n\nOriginal TEST DATA: \n");
    // for (int i = 0; i < h; i++) {
    //     for (int j = 0; j < w; j++) {
    //         printf("%d ", input[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n\n\n");




    // :::TESTS:::

    printf("\n\n\n::::::::Running basic convolution test::::::::\n\n");
    clock_t begin = clock();

    mock_FlagsDPack._convoTypeFlag = 0;
    int** basic_convo_res = covolveWrapper(mock_ImgDPack, mock_KerDPack, mock_FlagsDPack);

    clock_t end = clock();

    double runtime_used = (double)(end - begin) / CLOCKS_PER_SEC;



    // printf("\n\nBasic Convo result: \n");
    // for (int i = 0; i < h; i++) {
    //     for (int j = 0; j < w; j++) {
    //         printf("%d ", basic_convo_res[i][j]);
    //     }
    //     printf("\n");
    // }



    printf("\n\n\n::::::::Algo took %fs to finalize::::::::\n\n", runtime_used);

    printf("\n\n\n::::::::Running separated convolution test::::::::\n\n");
    begin = clock();

    mock_FlagsDPack._convoTypeFlag = 1;
    int** separated_convo_res = covolveWrapper(mock_ImgDPack, mock_KerDPack, mock_FlagsDPack);

    end = clock();

    runtime_used = (double)(end - begin) / CLOCKS_PER_SEC;



    // printf("\n\nSeparated Convo result: \n");
    // for (int i = 0; i < h; i++) {
    //     for (int j = 0; j < w; j++) {
    //         printf("%d ", separated_convo_res[i][j]);
    //     }
    //     printf("\n");
    // }



    printf("\n\n\n::::::::Algo took %fs to finalize::::::::\n\n", runtime_used);
    
     // !:::TESTS:::



    return 0;
}