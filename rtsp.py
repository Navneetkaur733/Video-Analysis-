import datetime
import cv2

VID_SIZE = 0.5
cam_name = 'Home'
root = r'C:\Zdaly\Zdaly-Vedio-Analysis\res\rec'
rtsp = "rtsp://admin:L2B4457B@192.168.18.74:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"

while(True):
    cap = cv2.VideoCapture(rtsp)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, frame = cap.read()
    height, width, _ = frame.shape
    start_time = datetime.datetime.now()

    fourCC = cv2.VideoWriter_fourcc(*'mp4v')
    vid_path = root + f'\{cam_name}\{start_time:%Y%m%d%H%M%S}.mp4'
    out = cv2.VideoWriter(vid_path , fourCC, fps, (width, height))

    while (datetime.datetime.now() - start_time) < datetime.timedelta(minutes=1):
        out.write(frame)
        frame = cv2.resize(frame, (int(width * VID_SIZE), int(height * VID_SIZE)))
        cv2.imshow('LIVE', frame)
        if cv2.waitKey(1) == 27: break;
        ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()