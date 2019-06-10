from numpy import *
from random import randint

def pic_to_arr(pic):
    #CONVERT TO 1D ARRAY

    row_pixels = array([])

    for row in pic:
        row_pixels = append(row_pixels,row)
        row_pixels = append(row_pixels,[2]) #new row signal
        print(row_pixels)

    pic = row_pixels
    #pad with end_pixel and new_row_pixel:
    pic = append([1,2],pic)
    pic= append(pic,[1])
    pic = array([pic])
    
    return pic

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


"""
assume knowledge of :
    SAMPLE_RATE
    FREQ
    T_END time of broadcast in seconds per pixel
"""
def demodulate_signal(recieved_signal,SAMPLE_RATE,T_END):
    
    time_axis=get_time_axis(recieved_signal,SAMPLE_RATE)
    
        #first pixel is end amp:
    cur_signal,recieved_signal = split(recieved_signal,[SAMPLE_RATE*T_END])
    cur_time_axis,time_axis=split(time_axis,[SAMPLE_RATE*T_END])

    fourier = fft.rfft(cur_signal)/SAMPLE_RATE*2
    end_pixel = amax(abs(fourier))
    print("end_pixel",end_pixel)

    #second pixel is new row pixel
    cur_signal,recieved_signal = split(recieved_signal,[SAMPLE_RATE*T_END])
    cur_time_axis,time_axis=split(time_axis,[SAMPLE_RATE*T_END])

    fourier = fft.rfft(cur_signal)/SAMPLE_RATE*2
    # freq = fft.rfftfreq()
    # fourier = 1*shift(fourier.real,-CARRIER_FREQ*T_END,cval=0)+1j*shift(fourier.imag,-CARRIER_FREQ*T_END,cval=0)
    new_row_pixel = amax(abs(fourier))
    # new_row_pixel = 2
    print("new_row_pixel",new_row_pixel)

    restored_2d = empty([0,0])
    restored_pixels = empty([0,0])


    # n_pixels = int(abs(size(recieved_signal)/SAMPLE_RATE))


    while(True):
        cur_signal,recieved_signal = split(recieved_signal,[SAMPLE_RATE*T_END])
        cur_time_axis,time_axis=split(time_axis,[SAMPLE_RATE*T_END])

        fourier = fft.rfft(cur_signal)/SAMPLE_RATE*2
    #     freq = fft.rfftfreq(cur_time_axis.shape[-1],1./SAMPLE_RATE)

        cur_pixel = amax(abs(fourier))

        if cur_pixel<1.5*end_pixel:
             print("reached end pixel")
             print(cur_pixel)             

             break

        elif cur_pixel<1.5*new_row_pixel:
            print("new row pixel")
            if restored_2d.size==0:
                restored_2d=restored_pixels
            else:
                restored_2d=append(restored_2d,restored_pixels,axis=0)

            print("restored_pixel shape",restored_pixels.shape)
            print("2d_pixel shape",restored_2d.shape)
            restored_pixels=empty([0,0])
            continue



        restored_pixels = append(restored_pixels, amax(abs(fourier))).reshape(1,restored_pixels.size+1)
        print(restored_pixels)             

    # for i in range(n_pixels):

    print("restored 2d is:",restored_2d)
    return restored_2d



    