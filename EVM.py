import cv2
import time
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
import queue

#convert RBG to YIQ
def rgb2ntsc(src):
    [rows,cols]=src.shape[:2]
    dst=np.zeros((rows,cols,3),dtype=np.float64)
    T = np.array([[0.114, 0.587, 0.298], [-0.321, -0.275, 0.596], [0.311, -0.528, 0.212]])
    for i in range(rows):
        for j in range(cols):
            dst[i, j]=np.dot(T,src[i,j])
    return dst

#convert YIQ to RBG
def ntsc2rbg(src):
    [rows, cols] = src.shape[:2]
    dst=np.zeros((rows,cols,3),dtype=np.float64)
    T = np.array([[1, -1.108, 1.705], [1, -0.272, -0.647], [1, 0.956, 0.620]])
    for i in range(rows):
        for j in range(cols):
            dst[i, j]=np.dot(T,src[i,j])
    return dst

#Build Gaussian Pyramid
def build_gaussian_pyramid(src,level=3):
    s=src.copy()
    pyramid=[s]
    for i in range(level):
        s=cv2.pyrDown(s)
        # print(s.shape)
        pyramid.append(s)
    return pyramid

#Build Laplacian Pyramid
def build_laplacian_pyramid(src,levels=3):
    gaussianPyramid = build_gaussian_pyramid(src, levels)
    print("这一帧的高斯处理完毕")
    pyramid=[]
    for i in range(levels,0,-1):
        GE=cv2.pyrUp(gaussianPyramid[i])
        # print(GE.shape)
        # print(gaussianPyramid[i-1].shape)
        L=cv2.subtract(gaussianPyramid[i-1],GE)
        pyramid.append(L)
    print("这一帧的拉普拉斯处理完毕")
    return pyramid

#load video from file
def load_video(video_filename):
    cap=cv2.VideoCapture(video_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 100
    print("这段视频有%d帧"%(frame_count))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width,height)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("这段视频的帧率是%d"%(fps))
    video_tensor=np.zeros((count,height,width),dtype='float')
    x=0
    while cap.isOpened():
        ret,frame=cap.read()
        count -= 1
        if ret is True and count > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将捕获的一帧图像灰度化处理
            video_tensor[x]=gray
            x+=1
        else:
            break
    print(x)
    return video_tensor,fps

# apply temporal ideal bandpass filter to gaussian video
def temporal_ideal_filter(tensor,low,high,fps,axis=0):
    fft=fftpack.fft(tensor,axis=axis)
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff=fftpack.ifft(fft, axis=axis)
    return np.abs(iff)

# build gaussian pyramid for video
def gaussian_video(video_tensor,levels=3):
    for i in range(0,video_tensor.shape[0]):
        frame=video_tensor[i]
        pyr=build_gaussian_pyramid(frame,level=levels)
        gaussian_frame=pyr[-1]
        if i==0:
            vid_data=np.zeros((video_tensor.shape[0],gaussian_frame.shape[0],gaussian_frame.shape[1]))
        vid_data[i]=gaussian_frame
    return vid_data

#amplify the video
def amplify_video(gaussian_vid,amplification=50):
    return gaussian_vid*amplification

#reconstract video from original video and gaussian video
def reconstract_video(amp_video,origin_video,levels=3):
    final_video=np.zeros(origin_video.shape)
    for i in range(0,amp_video.shape[0]):
        img = amp_video[i]
        for x in range(levels):
            img=cv2.pyrUp(img)
        img=img+origin_video[i]
        final_video[i]=img
    return final_video

#save video to files
def save_video(video_tensor):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    [height,width]=video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter("out.avi", fourcc, 30, (width, height), 1)
    for i in range(0,video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i]))
    writer.release()

#magnify color
def magnify_color(video_name,low,high,levels=3,amplification=10):
    t,f=load_video(video_name)
    gau_video=gaussian_video(t,levels=levels)
    filtered_tensor=temporal_ideal_filter(gau_video,low,high,f)
    amplified_video=amplify_video(filtered_tensor,amplification=amplification)
    final=reconstract_video(amplified_video,t,levels=3)
    print("视频重建完成")
    for x in range(3):
        for i in range(final.shape[0]):
            cv2.imshow("欧拉放大后的视频", final[i])
            cv2.waitKey(int(1000 / int(f)))  # 设置延迟时间
        cv2.destroyAllWindows()

#build laplacian pyramid for video
def laplacian_video(video_tensor,levels=3):
    tensor_list=[]
    for i in range(0,video_tensor.shape[0]):
        frame=video_tensor[i]
        print("取出第%d帧进行拉普拉斯金字塔处理"%(i))
        pyr=build_laplacian_pyramid(frame,levels=levels)
        if i==0:
            for k in range(levels):
                tensor_list.append(np.zeros((video_tensor.shape[0],pyr[k].shape[0],pyr[k].shape[1])))
        for n in range(levels):
            tensor_list[n][i] = pyr[n]
        print("第%d帧的拉普拉斯金字塔被保存到tensor_list里"%(i))
    return tensor_list

#butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    omega = 0.5 * fs
    low = lowcut / omega
    high = highcut / omega
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data, axis=0)
    return y

#reconstract video from laplacian pyramid
def reconstract_from_tensorlist(filter_tensor_list,levels=3):
    final=np.zeros(filter_tensor_list[-1].shape)
    # print(final)
    for i in range(filter_tensor_list[0].shape[0]):
        up = filter_tensor_list[0][i]
        for n in range(levels-1):
            up=cv2.pyrUp(up)+filter_tensor_list[n + 1][i]#可以改为up=cv2.pyrUp(up)
        final[i]=up
    return final

#manify motion
def magnify_motion(video_name,low,high,levels=3,amplification=30):
    t,f=load_video(video_name)
    print("视频已加载")
    lap_video_list=laplacian_video(t,levels=levels)
    print("所有帧的拉普拉斯金字塔图像保存完成")
    filter_tensor_list=[]
    for i in range(levels):
        print("第%d层拉普拉斯金字塔序列进行滤波处理"%(i))
        filter_tensor=butter_bandpass_filter(lap_video_list[i],low,high,f)
        print("第%d层拉普拉斯金字塔序列滤波完成，进行放大" % (i))
        filter_tensor*=amplification
        print("第%d层拉普拉斯金字塔序列滤波放大完成" % (i))
        filter_tensor_list.append(filter_tensor)
    print("拉普拉斯金字塔滤波放大完成，进行视频重建")
    recon=reconstract_from_tensorlist(filter_tensor_list,levels)
    final=t+recon
    data = queue.Queue()
    H = np.zeros(final[0].shape)
    yu = 100
    delta = 125
    print("视频重建完成")
    for x in range(3):
        for i in range(final.shape[0]):
            if i == 0:
                data.put(final[i])
                continue
            data.put(final[i])
            old_frame = data.get()
            a = cv2.addWeighted(old_frame.astype(float), 1, final[i].astype(float),
                                -1, 0)
            D = np.fabs(a)
            Psi = D >= yu
            # c = H - delta
            # H = np.maximum(0, c)
            H[Psi] = 250
            cv2.imshow("result", H.astype("uint8"))
            cv2.waitKey(int(1000 / int(f)))  # 设置延迟时间
            H = np.zeros(final[0].shape)
    cv2.destroyAllWindows()

if __name__=="__main__":
    # magnify_color("baby.mp4",0.4,3)
    magnify_motion("breath.mp4",0.4,3)
