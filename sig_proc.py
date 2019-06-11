from numpy import *
from random import randint
from numpy.fft import *


def pic_to_arr(pic):
    # generate prefix
    nrows,ncols = pic.shape

    # generate prefix in format BLACK,ncols*WHITE,BLACK,nrows*WHITE,BLACK

    prefix = array([])

    prefix = append(prefix,255)
    prefix = append(prefix,ones(ncols))
    prefix = append(prefix,255)
    prefix = append(prefix,ones(nrows))
    prefix = append(prefix,255)
    
    #generate pic_arr
    row_pixels = array([])
    for row in pic:
        row_pixels = append(row_pixels,row)

    
    #concat and return
    row_pixels = append(prefix,row_pixels)
    row_pixels = reshape(row_pixels,(1,size(row_pixels)))
    return row_pixels


    

def  modulate_arr(pic_arr,FREQ,TIME_PER_PIXEL,SAMPLE_RATE):
    t = arange(0,TIME_PER_PIXEL,1./SAMPLE_RATE)
    signal = array([])

    for i in pic_arr:
        for j in i:    
            signal = append(signal,j*sin(2*pi*FREQ*t))
    
    return signal

def add_noise(signal,NOISE_AMP):
    noise = [NOISE_AMP*randint(-50,50)/100 for i in signal]
    return signal+noise

def get_time_axis(signal,SAMPLE_RATE):
    return arange(0,size(signal)/SAMPLE_RATE,1/SAMPLE_RATE)





def process_psig(signal,SAMPLE_RATE,TPP):
    cur_psig, r_signal = split(signal,[int(SAMPLE_RATE*TPP)])
    fourier = rfft(cur_psig)/SAMPLE_RATE*2
    cur_pamp = amax(abs(fourier))
#     print("amp",cur_pamp)
    
    return r_signal,cur_pamp


def get_dimensions(signal,SAMPLE_RATE,TPP):

    r_signal,counter_amp = process_psig(signal,SAMPLE_RATE,TPP)

    ncols,nrows = (0,0)
    
    #count number of cols
    while(True):
        r_signal,cur_pamp = process_psig(r_signal,SAMPLE_RATE,TPP)
        if (cur_pamp>1./2*counter_amp): 
#             print("here",cur_pamp,1./2*counter_amp)
            break
        ncols += 1
    
#     print(ncols)
    
    while(True):
        r_signal,cur_pamp = process_psig(r_signal,SAMPLE_RATE,TPP)
        if (cur_pamp>1./2*counter_amp): 
#             print(cur_pamp,2*counter_amp)
            break
        nrows += 1

#     print(nrows)
    print("DIM=",nrows,ncols)
    
    return ncols,nrows,r_signal

    
#     ncols,nrows,r_signal = get_dimensions(signal,SAMPLE_RATE,TPP)


def process_row(r_signal,SAMPLE_RATE,TPP,ncols):
    
    row = array([])
    for i in range(ncols):
        r_signal,cur_pamp = process_psig(r_signal,SAMPLE_RATE,TPP)

        row = append(row,cur_pamp)
#         print(row)
    row = reshape(row,(1,size(row)))
#     print('row shape:',row.shape)
    return row,r_signal 
    
# row,r_signal=process_row(r_signal,SAMPLE_RATE,TPP,ncols)


def process_signal(signal,SAMPLE_RATE,TPP):
    
    ncols,nrows,r_signal = get_dimensions(signal,SAMPLE_RATE,TPP)
    
    restored_pic,r_signal = process_row(r_signal,SAMPLE_RATE,TPP,ncols)
#     restored_pic = restored_pic.reshape(1,ncols)
#     print('shape',restored_pic.shape)

    for r in range(nrows-1):
        row,r_signal=process_row(r_signal,SAMPLE_RATE,TPP,ncols)
        
        
        restored_pic = append(restored_pic,row,axis=0)
        
#         print('shape',restored_pic.shape)
    return restored_pic

    # signal = loadtxt(open(SIGNAL_FILE))
    # process_signal(signal,SAMPLE_RATE,TPP)
    