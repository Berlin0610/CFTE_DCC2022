import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator ###
from modules.keypoint_detector import KPDetector ###
#from modules.model import GeneratorFullModel, DiscriminatorFullModel
from modules.dists import *
from modules.util import *
from animate import normalize_kp
from scipy.spatial import ConvexHull
import scipy.io as io
import json
import cv2
import torch.nn.functional as F
import struct, time
from pathlib import Path
# new added start
import compressai
from compressai.zoo import models
# new added end
from compressai.models import CompressionModel
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from arithmetic.value_encoder import *
from arithmetic.value_decoder import *
# from torchsummaryX import summary


def rgb4442yuv420(dir1, dir2):    
    os.makedirs(dir2,exist_ok=True)
    files=glob.glob(os.path.join(dir1,'*.rgb'))
    files.sort()
    for file in files:  
        f=open(file,'rb')
        file_name=file.split('/')[-1]
        file_name=os.path.splitext(file_name)[0]
        tar_path=dir2+file_name+'.yuv'
        yuvfile=open(tar_path, mode='w')

        width=256
        height=256
        framenum=250
        for idx in range(framenum):
            R=np.fromfile(f,np.uint8,width*height).reshape((height,width))
            G=np.fromfile(f,np.uint8,width*height).reshape((height,width))   
            B=np.fromfile(f,np.uint8,width*height).reshape((height,width))
            image_rgb = np.zeros((height,width,3), 'uint8')
            image_rgb[..., 0] = R
            image_rgb[..., 1] = G
            image_rgb[..., 2] = B
            image_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)
            image_yuv[:,:,0].tofile(yuvfile)
            image_yuv[:,:,1].tofile(yuvfile)
            image_yuv[:,:,2].tofile(yuvfile)
        
    print('done')


def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        #config = yaml.load(f)
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'],strict=False)
    kp_detector.load_state_dict(checkpoint['kp_detector']) ####

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
        
    generator.eval()
    kp_detector.eval()
    return generator, kp_detector


def make_prediction(reference_frame, kp_reference, kp_current, generator, relative=False, adapt_movement_scale=False, cpu=False):
        
    kp_norm = normalize_kp(kp_source=kp_reference, kp_driving=kp_current,
                           kp_driving_initial=kp_reference, use_relative_movement=relative,
                           use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
    
    out = generator(reference_frame, kp_reference, kp_norm)
    # summary(generator, reference_frame, kp_reference, kp_norm)
    
    prediction=np.transpose(out['prediction'].data.cpu().numpy(), [0, 1, 2, 3])[0]

    return prediction

def check_reference(ref_kp_list, kp_current):
    diff_list=[]
    for idx in range(0, len(ref_kp_list)):    
        dif = (ref_kp_list[idx]['value'] - kp_current['value']).abs().mean() 
        diff_list.append(dif)
    return diff_list

    
if __name__ == "__main__":
    parser = ArgumentParser()
            


    config_path='./checkpoint/vox-256/vox-256.yaml'
    checkpoint='./checkpoint/vox-256/00000099-checkpoint.pth.tar'            
    generator, kp_detector = load_checkpoints(config_path, checkpoint, cpu=False) 

    frames=250
    width=256
    height=256
    Qstep=4

    seqlist=['001']       
    qplist=['32','37','42','47','52']   

    modeldir = 'CFTE' 
    model_dirname='./experiment/'+modeldir+"/"     


    totalResult=np.zeros((len(seqlist)+1,len(qplist)))
    for seqIdx, seq in enumerate(seqlist):
        for qpIdx, QP in enumerate(qplist):             

            driving_kp =model_dirname+'/kp/'+str(width)+'/'+seq+'_QP'+str(QP)+'/'   
            dir_dec=model_dirname+'/dec/'+str(width)+'/'   
            os.makedirs(dir_dec,exist_ok=True)     # the real decoded video  
            decode_seq=dir_dec+seq+'_QP'+str(QP)+'.rgb'

            dir_enc =model_dirname+'/enc/'+str(width)+'/'+seq+'_QP'+str(QP)+'/'
            os.makedirs(dir_enc,exist_ok=True)     # the frames to be compressed by vtm     

            savetxt=model_dirname+'/resultBit/'+str(width)+'/'
            os.makedirs(savetxt,exist_ok=True)         

            f_dec=open(decode_seq,'w') 
            seq_kp_integer=[]

            start=time.time() 
            gene_time = 0
            fusion_time = 0
            sum_bits = 0

            for frame_idx in range(0, frames):            

                frame_idx_str = str(frame_idx).zfill(4)   

                if frame_idx in [0]:      # I-frame                      
                    os.system("./vtm/decode.sh "+dir_enc+'frame'+frame_idx_str)

                    bin_file=dir_enc+'frame'+frame_idx_str+'.bin'
                    bits=os.path.getsize(bin_file)*8
                    sum_bits += bits

                    f_temp=open(dir_enc+'frame'+frame_idx_str+'_dec.rgb','rb')
                    img_rec=np.fromfile(f_temp,np.uint8,3*height*width).reshape((3,height,width))   # 3xHxW RGB         
                    img_rec.tofile(f_dec) 

                    img_rec = resize(img_rec, (3, height, width))    # normlize to 0-1                                      
                    with torch.no_grad(): 
                        reference = torch.tensor(img_rec[np.newaxis].astype(np.float32))
                        reference = reference.cuda()    # require GPU
                        kp_reference = kp_detector(reference) ################
                        kp_value = kp_reference['value']
                        kp_value_list = kp_value.tolist()
                        kp_value_list = str(kp_value_list)
                        kp_value_list = "".join(kp_value_list.split())

                        kp_value_frame=json.loads(kp_value_list)
                        kp_value_frame= eval('[%s]'%repr(kp_value_frame).replace('[', '').replace(']', ''))
                        seq_kp_integer.append(kp_value_frame)                                    


                else:

                    frame_index=str(frame_idx).zfill(4)
                    bin_save=driving_kp+'/frame'+frame_index+'.bin'            
                    kp_dec = final_decoder_expgolomb(bin_save)

                    ## decoding residual
                    kp_difference = data_convert_inverse_expgolomb(kp_dec)
                    ## inverse quanzation
                    kp_difference_dec=[i/Qstep for i in kp_difference]
                    kp_difference_dec= eval('[%s]'%repr(kp_difference_dec).replace('[', '').replace(']', ''))  

                    kp_previous=seq_kp_integer[frame_idx-1]


                    kp_previous= eval('[%s]'%repr(kp_previous).replace('[', '').replace(']', '').replace("'", ""))  

                    kp_integer=listformat_adptive(kp_previous, kp_difference_dec, 1,4)  #####                        

                    seq_kp_integer.append(kp_integer)

                    kp_integer=json.loads(str(kp_integer))
                    kp_current_value=torch.Tensor(kp_integer).to('cuda:0')          
                    dict={}
                    dict['value']=kp_current_value  
                    kp_current=dict 

                    gene_start = time.time()
                    prediction = make_prediction(reference, kp_reference, kp_current, generator) #######################
                    gene_end = time.time()
                    gene_time += gene_end - gene_start
                    pre=(prediction*255).astype(np.uint8)  
                    pre.tofile(f_dec)                              

                    ###
                    frame_index=str(frame_idx).zfill(4)
                    bin_save=driving_kp+'/frame'+frame_index+'.bin'
                    bits=os.path.getsize(bin_save)*8
                    sum_bits += bits



            f_dec.close()     

            end=time.time()

            totalResult[seqIdx][qpIdx]=sum_bits   

            print(seq+'_QP'+str(QP)+'.rgb',"success. Total time is %.4fs. Model inference time is %.4fs. Total bits are %d" %(end-start,gene_time,sum_bits))

    # summary the bitrate
    for qp in range(len(qplist)):
        for seq in range(len(seqlist)):
            totalResult[-1][qp]+=totalResult[seq][qp]
        totalResult[-1][qp] /= len(seqlist)

    np.set_printoptions(precision=5)
    totalResult = totalResult/1000
    seqlength = frames/25
    totalResult = totalResult/seqlength

    np.savetxt(savetxt+'/resultBit.txt', totalResult, fmt = '%.5f')            


