# Author: Mr.Wang
# CreateTime: 2021/11/10
# FileName: lianxi
#FUNCTION:
import numpy as np
import cv2
import queue

class MHI:
    def __init__(self,cap,tau,delta,xi,t):
        self.tau=tau
        self.delta=delta
        self.xi=xi
        self.t=t
        self.cap=cap
        self.data = queue.Queue()
        ret,frame=cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            print(frame.shape)
            for i in range(t):
                self.data.put(frame)
        self.H = np.zeros(frame.shape)
    def getimag(self):
        ret,frame=cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            return ret,frame
        self.data.put(frame)
        old_frame=self.data.get()
        a=cv2.addWeighted(old_frame.astype(float),1, frame.astype(float), -1, 0)
        D= np.fabs(a)
        Psi= D >=self.xi
        # px = np.sum(frame[Psi])
        # print(px)
        c=self.H-self.delta
        H=np.maximum(0,c)
        H[Psi]=self.tau
        p = np.sum(H)
        print(p)
        self.H=H
        return ret, H.astype("uint8")


cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(fps)
a=MHI(cap,tau=255,xi=20,delta=25,t=1)
while True:
    _,frame=a.getimag()
    cv2.imshow("out_win", frame)
    # cv2.waitKey(int(1000 / int(fps)))  # 设置延迟时间
    if cv2.waitKey(1) & 0xFF == 27:
        cap.release()
        cv2.destroyAllWindows()