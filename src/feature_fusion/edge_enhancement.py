import cv2
import numpy as np
import scipy.signal as sp
import src.feature_fusion.generators as g
from src.types.defines import global_vars
import ctypes
import platform
import os

class t_imageDPack(ctypes.Structure):
    """ creates a struct for C data type "t_imageDPack"  """
    _fields_ = [('_inputImageMat', ctypes.POINTER(ctypes.POINTER(ctypes.c_int))),
                ('width', ctypes.c_int),
                ('height', ctypes.c_int)] 

class t_kernelDPack(ctypes.Structure):
    """ creates a struct for C data type "t_kernelDPack"  """
    _fields_ = [('_kernel2D', ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
                ('_kernelHorizontal', ctypes.POINTER(ctypes.c_double)),
                ('_kernelVertical', ctypes.POINTER(ctypes.c_double)),
                ('kernelSize', ctypes.c_int)] 

class t_flagsDPack(ctypes.Structure):
    """ creates a struct for C data type "t_flagsDPack"  """
    _fields_ = [('_paddingFlag', ctypes.c_int),
                ('_convoTypeFlag', ctypes.c_int)] 


def setup_c_lib():
    if platform.system() == 'Windows':
        c_lib = ctypes.CDLL(os.path.abspath('src/optimz/dynamic_libs/convo_lib.dll'))
    elif platform.system() == 'Linux':
        c_lib = ctypes.CDLL(os.path.abspath('src/optimz/dynamic_libs/convo_lib.so'))
    elif platform.system() == 'Darwin':
        c_lib = ctypes.CDLL(os.path.abspath('src/optimz/dynamic_libs/convo_lib.dylib'))
    else:
        c_lib = ctypes.CDLL(os.path.abspath('src/optimz/dynamic_libs/convo_lib.so'))

    # Section C convolution wrapper
    convol_c = c_lib.covolveWrapper

    convol_c.argtypes = [t_imageDPack, t_kernelDPack, t_flagsDPack]
    convol_c.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))

    return convol_c

def edge_enhancement(src, blurred):
    E = (src - global_vars["weight"] * blurred) / (1 - global_vars["weight"])

    E = np.clip(E, 0, 255).astype(np.uint8)

    return E

def edge_enhancement_wrapper(src):
    ch_0, ch_1, ch_2 = cv2.split(src)
    imHeight, imWidth, imCh = src.shape

    fDPack = t_flagsDPack()
    imgDPack_ch0, imgDPack_ch1, imgDPack_ch2 = t_imageDPack(), t_imageDPack(), t_imageDPack()
    kDPack = t_kernelDPack()

    k2D=g.GK_generator(1)

    try:
        my_convol = setup_c_lib()
        cLib_flag = 1
    except: 
        print("Warning: Failed to load the dynamic library")    
        my_convol = None
  
    if(my_convol == None):
        print("Warning: C created lib failed to set up due to internal errors!\n\n")
        print("Continuing with scipy built in convolution function")
        cLib_flag = 0

    if(np.isclose(np.linalg.matrix_rank(k2D), 1, atol=.5)):
        k_horiz, k_vert = g.GK_separator(k2D)
        
        k_reconstr = np.outer(k_horiz, k_vert)
        if(k_reconstr.all() == k2D.all()):
            print("\n\nKernel separated succesfully!\n\n")
        else:
            print("\n\nKernel separation error occured!\n\n")


    if(cLib_flag == 0):
        blurred_img_ch0 = sp.convolve2d(src[:,:,0], k2D, boundary='symm', mode='same')
        blurred_img_ch1 = sp.convolve2d(src[:,:,1], k2D, boundary='symm', mode='same')
        blurred_img_ch2 = sp.convolve2d(src[:,:,2], k2D, boundary='symm', mode='same')

        blurred_img = cv2.merge([blurred_img_ch0, blurred_img_ch1, blurred_img_ch2]).astype(np.float32)
    else:
        # Pack input data for C written convolution
        ch0_int32 = ch_0.astype(np.int32)
        ch0_int32_ptr = (ctypes.POINTER(ctypes.c_int) * imHeight)()
        for i in range(imHeight):
            ch0_int32_ptr[i] = ch0_int32[i].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        imgDPack_ch0._inputImageMat = ch0_int32_ptr
        imgDPack_ch0.width = ctypes.c_int(imWidth)
        imgDPack_ch0.height = ctypes.c_int(imHeight)

        ch1_int32 = ch_1.astype(np.int32)
        ch1_int32_ptr = (ctypes.POINTER(ctypes.c_int) * imHeight)()
        for i in range(imHeight):
            ch1_int32_ptr[i] = ch1_int32[i].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        imgDPack_ch1._inputImageMat = ch1_int32_ptr
        imgDPack_ch1.width = ctypes.c_int(imWidth)
        imgDPack_ch1.height = ctypes.c_int(imHeight)

        ch2_int32 = ch_2.astype(np.int32)
        ch2_int32_ptr = (ctypes.POINTER(ctypes.c_int) * imHeight)()
        for i in range(imHeight):
            ch2_int32_ptr[i] = ch2_int32[i].ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        imgDPack_ch2._inputImageMat = ch2_int32_ptr
        imgDPack_ch2.width = ctypes.c_int(imWidth)
        imgDPack_ch2.height = ctypes.c_int(imHeight)

        kDPack._kernel2D = (ctypes.POINTER(ctypes.c_double) * global_vars["kernel_size"])(*[kRow.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for kRow in k2D])
        kDPack._kernelHorizontal = k_horiz.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        kDPack._kernelVertical = k_vert.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        kDPack.kernelSize = ctypes.c_int(global_vars["kernel_size"])
        
        fDPack._paddingFlag = ctypes.c_int(global_vars["padding_flag"]) # 1 -replicate-padding | 0 - zero-padding
        fDPack._convoTypeFlag = ctypes.c_int(1) # 1 - separated convolution | 0 - basic convolution

        packed_img_ch0 = my_convol(imgDPack_ch0, kDPack, fDPack)
        packed_img_ch1 = my_convol(imgDPack_ch1, kDPack, fDPack)
        packed_img_ch2 = my_convol(imgDPack_ch2, kDPack, fDPack)

        unpacked_img_ch0, unpacked_img_ch1, unpacked_img_ch2 = [],[],[]
        for i in range(imHeight):
            rptr_ch0 = packed_img_ch0[i]
            rptr_ch1 = packed_img_ch1[i]
            rptr_ch2 = packed_img_ch2[i]
            
            rdata_ch0 = [rptr_ch0[j] for j in range(imWidth)]
            rdata_ch1 = [rptr_ch1[j] for j in range(imWidth)]
            rdata_ch2 = [rptr_ch2[j] for j in range(imWidth)]

            unpacked_img_ch0.append(rdata_ch0)
            unpacked_img_ch1.append(rdata_ch1)
            unpacked_img_ch2.append(rdata_ch2)

        # uint8(int)
        ch0_res = np.array(unpacked_img_ch0, dtype=np.uint8)
        ch1_res = np.array(unpacked_img_ch1, dtype=np.uint8)
        ch2_res = np.array(unpacked_img_ch2, dtype=np.uint8)

        # float32(float)
        # ch0_res = np.array(unpacked_img_ch0, dtype=np.float32)
        # ch1_res = np.array(unpacked_img_ch1, dtype=np.float32)
        # ch2_res = np.array(unpacked_img_ch2, dtype=np.float32)

        blurred_img = cv2.merge([ch0_res, ch1_res, ch2_res])

        if(blurred_img.any() == None):
            print("Warning: C created lib failed to perform the convolution\n")
            print("Continuing with scipy built in convolution function\n\n")
            
            blurred_img_ch0 = sp.convolve2d(src[:,:,0], k2D, boundary='symm', mode='same')
            blurred_img_ch1 = sp.convolve2d(src[:,:,1], k2D, boundary='symm', mode='same')
            blurred_img_ch2 = sp.convolve2d(src[:,:,2], k2D, boundary='symm', mode='same')
            
            blurred_img = cv2.merge([blurred_img_ch0, blurred_img_ch1, blurred_img_ch2]).astype(np.float32)

    enhanced_img=edge_enhancement(src, blurred_img)

    enhanced_img = np.clip(enhanced_img, 0, 255).astype(np.uint8)

    return enhanced_img


