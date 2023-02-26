import pandas as pd
import numpy as np
import statistics
import datetime
import cv2
import os

from reader import OCR
import anpr

PIXEL_COUNT = 255
COLOR_SCHEME = 'BGR'
MODEL_INPUT_SIZE = 416, 416

CFG = r'C:\Zdaly\Zdaly-Vedio-Analysis\res\yolo\yolov4.cfg'
LABELS = r'C:\Zdaly\Zdaly-Vedio-Analysis\res\yolo\labels.txt'
WEIGHTS = r'C:\Zdaly\Zdaly-Vedio-Analysis\res\yolo\yolov4.weights'

CONFIDENCE_THRESHOLD = 0.2
FRAME_TRACKING_DIST = 50
NMS_THRESHOLD = 0.4

BORDER_THICKNESS = 2
WHITE = 255, 255, 255
YELLOW = 0, 255, 255
GREEN = 0, 255, 0
BLUE = 255, 0, 0
RED = 0, 0, 255
BLACK = 0, 0, 0

FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE_TYPE = cv2.LINE_AA
VID_SCALE = 0.50
FONT_SCALE = 1
VID_SPEED = 2

DETECTION_AREA = 0.2
EROSION_KERNEL = np.ones((3, 3), np.uint8)

GOAL = {'person', 'car', 'moterbike', 'bus', 'truck'}


def draw_lines(vid_path, video=True):
    if video:
        cap = cv2.VideoCapture(vid_path)
        ret, img = cap.read()
        if not ret:
            print('Invalid Video')
            exit(1)
    else:
        img = cv2.imread(vid_path)

    counting_lines = []
    starting, ending = None, None
    def __callback__(event, x, y, _, __):
        global starting, ending
        if event == cv2.EVENT_LBUTTONDOWN:
            starting = (x,y)
        elif event == cv2.EVENT_LBUTTONUP:
            if starting is None: return;
            ending = (x,y)
            dist = distance(starting[0], ending[0], starting[1], ending[1])
            if dist <= FRAME_TRACKING_DIST: return;
            cv2.line(img, starting, ending, YELLOW, BORDER_THICKNESS)
            cv2.imshow('Draw Crossing Lines', img)
            counting_lines.append((starting, ending))

    cv2.namedWindow('Draw Crossing Lines')
    cv2.setMouseCallback('Draw Crossing Lines', __callback__)
    cv2.imshow('Draw Crossing Lines', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return np.array(counting_lines, np.uint32)

def draw_text(img, text, font=cv2.FONT_HERSHEY_PLAIN, pos=(0, 0), font_scale=3, font_thickness=2,
          text_color=(0, 255, 0), text_color_bg=(0, 0, 0)):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w + font_thickness*3, y - text_h - font_thickness*3), text_color_bg, -1)
    cv2.putText(img, text, (x, y-font_thickness), font, font_scale, text_color, font_thickness)

def get_bounding_box_for_line(line, h1, mh, mw, delta=None):
    (x1, y1),(x2, y2)=line
    delta =  y2 - y1 if delta is None else delta
    rx1, ry1 = min(x1 + delta, mw), max(y1 - h1, 0)
    rx2, ry2 = min(x2 + delta, mw), max(y2 - h1, 0)
    rx3, ry3 = max(x1 - delta, 0), min(y1 + h1, mh)
    rx4, ry4 = max(x2 - delta, 0), min(y2 + h1, mh)
    return np.array([[rx1, ry1],[rx2, ry2],[rx4, ry4],[rx3, ry3]], np.int32).reshape((-1, 1, 2))

def get_id_for_cp(cp, v_map, id_counter):
    for k, vs in v_map.items():
        v = vs[-1]
        dist = distance(cp[0], v[0], cp[1], v[1])
        if dist > FRAME_TRACKING_DIST: continue;
        v_map[k] = vs + [cp]
        return k, v_map, id_counter

    id_counter += 1
    v_map[id_counter] = [cp]
    return id_counter, v_map, id_counter

distance = lambda x1, x2, y1, y2: ((x1 - x2)**2 + (y1 - y2)**2)**(1/2)

class VehicleCounter:

    def __init__(self, vid, weights=WEIGHTS, cfg=CFG, labels=LABELS, model_inp_size=MODEL_INPUT_SIZE,
                    model_scale=1/PIXEL_COUNT, color_scheme= COLOR_SCHEME, counting_lines=[]):

        if not os.path.exists(weights):
            raise FileNotFoundError('Weights not Found at', weights)

        if not os.path.exists(cfg):
            raise FileNotFoundError('Configuration not Found at', cfg)

        if not os.path.exists(labels):
            raise FileNotFoundError('Labels not Found at', labels)

        if type(vid) == str and not os.path.exists(weights):
            raise FileNotFoundError('Video not Found at', vid)

        net = cv2.dnn.readNet(weights, cfg)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=model_inp_size, scale=model_scale, swapRB=color_scheme=='BGR')

        self.labels = []
        with open(labels, 'r') as file:
            self.labels = file.read().split('\n')

        self.vid = vid
        self.vid_path = vid if type(vid) == str else ''
        self.cap = cv2.VideoCapture(vid)
        self.counting_lines = counting_lines
        self.line_counts = [0 for _ in range(len(counting_lines))]
        self.boxes = []

        self.vehicle_map = dict()
        self.crossed_ids = dict()
        self.car_info = dict()
        self.id_counter = 0

        self.frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.video_time = datetime.timedelta(seconds=int(self.frame_count / self.fps))

        self.reader = OCR()
        self.display = False
        self.goal = GOAL
        self.api_calls = 0
        self.conf = CONFIDENCE_THRESHOLD
        self.vid_speed = VID_SPEED
        self.roi = None

        self.df = pd.DataFrame(columns=['s_id', 'client', 'c_id', 'time', 'frame', 'o_id', 'label', 'x',
        'y', 'w', 'h', 'score', 'l_id', 'b_id'])


    def __update_cp__(self, cp, frame_dist):
        for k, v in self.vehicle_map.items():
            dist = distance(cp[0], v[0], cp[1], v[1])
            if dist > frame_dist: continue;
            self.vehicle_map[k] = cp
            return k
        self.id_counter += 1
        self.vehicle_map[self.id_counter] = cp
        return self.id_counter

    def __is_crossing__(self, box):
        if not len(self.counting_lines): return None;

        x1, y1, w, h = box
        x2, y2, x3, y3, x4, y4 = x1 + w, y1, x1, y1 + h, x1 + w, y1 + h
        box_lines = [((x1, y1), (x3, y3)), ((x3, y3), (x4, y4)), ((x4, y4), (x2, y2)), ((x2, y2), (x1, y1))]

        ccw = lambda x1, y1, x2, y2, x3, y3: (y3 - y1) * (x2 - x1) > (y2 - y1) * (x3-x1)
        intersect = lambda x1, y1, x2, y2, x3, y3, x4, y4: ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4)

        for idx, ((x1, y1), (x2, y2)) in enumerate(self.counting_lines):
            for ((x3, y3), (x4, y4)) in box_lines:
                if intersect(x1, y1, x2, y2, x3, y3, x4, y4):
                    return idx
        return None

    def __is_inside__(self, cp):
        if not len(self.boxes): return -1;
        for idx, (bx, by, bw, bh) in enumerate(self.boxes):
            if(bx <= cp[0] <= bx+bw) and (by <= cp[1] <= by+bh):
                return idx;
        return -1


    def __extract_info_from_frame__(self, img, nms, frame_dist, car_info, frame_count):
        frame = img.copy()
        curr_center_points = set()
        height, width, _ = img.shape
        get_center_point = lambda x, y, w, h: ((x + x + w)//2, (y + y + h)//2)

        curr_roi = frame[:,:]
        if self.roi is not None:
            x, y, w, h = self.roi
            curr_roi = curr_roi[y:min(y+h,height), x:min(x+w,width)]

        ls, ss, bbs = self.model.detect(curr_roi, self.conf, nms)

        for l, sc, (x1, y1, w, h) in zip(ls, ss, bbs):
            if self.labels[l] not in self.goal: continue;

            x, y = x1, y1
            if self.roi is not None:
                x, y = x1 + self.roi[0], y1 + self.roi[1]
            cp = get_center_point(x, y, w, h)
            box_idx = self.__is_inside__(cp)
            if box_idx < 0: continue;

            cp_id = self.__update_cp__(cp, frame_dist)
            curr_center_points.add(cp)

            if self.display:
                cv2.rectangle(img, (x, y, w, h), GREEN, BORDER_THICKNESS)
                draw_text(img, f'{self.labels[l]} {cp_id}', FONT, (x, y), FONT_SCALE, BORDER_THICKNESS, WHITE, GREEN)


            if cp_id not in self.crossed_ids:
                line_id = self.__is_crossing__((x, y, w, h))
                if line_id is None: continue
                self.line_counts[line_id] += 1
                self.crossed_ids[cp_id] = line_id

                if car_info and cp_id not in self.car_info:
                    self.api_calls += 1
                    info = self.reader.read_number_plate(frame[y:y+h, x:x+w])
                    # info = read_plates(frame[y:y+h, x:x+w])
                    if info:
                        self.car_info[cp_id] = info

            self.df = pd.concat([pd.DataFrame({
                    'time' : str(datetime.timedelta(seconds=frame_count//self.fps)),
                    'frame' : frame_count % self.fps,
                    'o_id' : cp_id,
                    'label' : self.labels[l],
                    'x' : x,
                    'y' : y,
                    'w' : w,
                    'h' : h,
                    'score' : sc,
                    'l_id': self.crossed_ids[cp_id],
                    'b_id': box_idx
                    }, index = [self.df.shape[0]]),
                 self.df]).reset_index(drop=True)

        return img, curr_center_points

    def __fit_line_in_box__(self, line_idx, box):
        x, y, w, h = cv2.boundingRect(box)
        line = self.counting_lines[line_idx]
        max_w = 2 * abs(line[0][0] - line[1][0])
        if w > max_w:
            x, w= x + ((w-max_w) // 2), max_w
        return x, y, w, h


    def __analyse_movement_by_line__(self, line_idx, area, duration):
        cap_cpy = cv2.VideoCapture(self.vid)
        bg_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=self.conf*100)

        _, img = cap_cpy.read()
        height, width, _ = img.shape

        xs, delta, box_areas = set(), None, set()
        line = self.counting_lines[line_idx]
        curr_frame_count, curr_seconds = 1, 0
        while(curr_seconds < duration):
            curr_seconds = curr_frame_count // self.fps
            cv2.putText(img, f'Analysing Line for {duration - curr_seconds} secs ...', (0, 50), FONT, FONT_SCALE, RED, BORDER_THICKNESS, LINE_TYPE)

            delta = int(statistics.median(list(xs)) - line[0][0]) if len(xs) else None
            box = get_bounding_box_for_line(line, int(height*area), height, width, delta=delta)


            x, y, w, h = self.__fit_line_in_box__(line_idx, box)
            fg = bg_sub.apply(img[y:min(y+h,height), x:min(x+w,width)]) #Subtracting BG
            _, fg = cv2.threshold(fg, PIXEL_COUNT-1, PIXEL_COUNT, cv2.THRESH_BINARY) #Keeping pure whites oly
            fg = cv2.erode(fg, EROSION_KERNEL, iterations=2)
            fg = cv2.dilate(fg, EROSION_KERNEL, iterations=2)
            contours, _ = cv2.findContours(fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                box_area = cv2.contourArea(cnt)
                if box_area < FRAME_TRACKING_DIST*10: continue;
                box_areas.add(box_area//(height*width))
                bx1, by1, bw, bh = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x+bx1, y+by1,bw, bh), RED, BORDER_THICKNESS)
                xs.add(x+bx1)


            starting_point, ending_point = line
            cv2.polylines(img, [box], True, YELLOW, BORDER_THICKNESS)
            cv2.rectangle(img, self.__fit_line_in_box__(line_idx, box), BLUE, BORDER_THICKNESS)
            cv2.line(img, starting_point, ending_point, YELLOW, BORDER_THICKNESS)

            img = cv2.resize(img, (int(width * VID_SCALE), int(height * VID_SCALE)))
            # cv2.imshow('roi',fg)
            cv2.imshow('Output', img)
            if cv2.waitKey(1) == 27: break;
            _, img = cap_cpy.read()
            curr_frame_count += 1

            curr_time = str(datetime.timedelta(seconds = curr_frame_count // self.fps))
            os.system('cls')
            print('Current Video Time    :', curr_time, curr_frame_count%self.fps)
            print('Total Video Time      :', self.video_time, self.fps)
            print('Line Index            :', line_idx+1, '/', len(self.counting_lines))
            print('BOX                   :', x, y, w, h)
            print('Delta                 :', delta)

        cap_cpy.release()
        self.boxes += [(x, y, w, h )]
        return box_areas

    def detect_lines(self, duration=10, stop=False):
        cap_cpy = cv2.VideoCapture(self.vid)
        bg_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=self.conf*100)
        get_center_point = lambda x, y, w, h: ((x + x + w)//2, (y + y + h)//2)

        _, img = cap_cpy.read()
        height, width, _ = img.shape
        curr_frame_count, curr_seconds = 1, 0
        duration = min(self.video_time.seconds, duration)
        v_map, id_counter = dict(), 0
        while(curr_seconds < duration):
            curr_seconds = curr_frame_count // self.fps

            fg = bg_sub.apply(img) #Subtracting BG
            _, fg = cv2.threshold(fg, PIXEL_COUNT-1, PIXEL_COUNT, cv2.THRESH_BINARY) #Keeping pure whites oly
            fg = cv2.erode(fg, EROSION_KERNEL, iterations=2)
            fg = cv2.dilate(fg, EROSION_KERNEL, iterations=2)

            cv2.putText(img, f'Detecting Line for {duration - curr_seconds} secs ...', (0, 50), FONT, FONT_SCALE, RED, BORDER_THICKNESS, LINE_TYPE)
            contours, _ = cv2.findContours(fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                box_area = cv2.contourArea(cnt)
                if box_area < FRAME_TRACKING_DIST*10: continue;
                bx1, by1, bw, bh = cv2.boundingRect(cnt)

                cp = get_center_point(bx1, by1, bw, bh)
                cp_id, v_map, id_counter = get_id_for_cp(cp, v_map, id_counter)

                cv2.rectangle(img, (bx1, by1, bw, bh), GREEN, BORDER_THICKNESS)
                cv2.circle(img, cp, BORDER_THICKNESS, RED, -1*BORDER_THICKNESS)
                # cv2.putText(img, str(cp_id), (cp[0], cp[1]), FONT, FONT_SCALE, RED, BORDER_THICKNESS, LINE_TYPE)

            img = cv2.resize(img, (int(width * VID_SCALE), int(height * VID_SCALE)))
            cv2.imshow('Output', img)
            if cv2.waitKey(1) == 27: break;
            ret, img = cap_cpy.read()
            curr_frame_count += 1



        _, img = cap_cpy.read()
        fg = bg_sub.apply(img) #Subtracting BG
        _, fg = cv2.threshold(fg, PIXEL_COUNT-1, PIXEL_COUNT, cv2.THRESH_BINARY) #Keeping pure whites oly
        fg = cv2.erode(fg, EROSION_KERNEL, iterations=2)
        fg = cv2.dilate(fg, EROSION_KERNEL, iterations=2)
        for cp_id, vs in v_map.items():
            for cp in vs:
                cv2.circle(fg, cp, BORDER_THICKNESS*10, WHITE, -1*BORDER_THICKNESS*10)#COLORS[(len(COLORS) - 1)%cp_id]
        _, fg = cv2.threshold(fg, PIXEL_COUNT-254, PIXEL_COUNT, cv2.THRESH_BINARY) #Keeping pure whites only

        lanes = []
        contours, _ = cv2.findContours(fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            box_area = cv2.contourArea(cnt)
            if box_area < FRAME_TRACKING_DIST*10: continue;
            box = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(int)
            lanes += [(box, box_area)]

        lanes = sorted(lanes, key=lambda x:x[-1], reverse=True)
        # pprint(lanes)
        thresh = lanes[0][-1] // 5
        lanes = [l for l in lanes if l[-1] > thresh]

        for (box, _) in lanes:
            min_d, line = float('inf'), None
            for i in range(4):
                for j in range(4):
                    if i != j:
                        curr_d = distance(box[i][0], box[j][0], box[i][1], box[j][1])
                        if curr_d < min_d:
                            min_d = curr_d
                            line = [box[i], box[j]]
            (x1, y1), (x2, y2) = line
            y_diff = (y2 - y1) // 2
            x_diff = (x2 - x1) // 2
            
            y_med = int(np.median([b[1] for b in box]))
            x_med = int(np.median([b[0] for b in box]))
            y_min, y_max = int(y_med - y_diff), int(y_med + y_diff)
            x_min, x_max = int(x_med - x_diff), int(x_med + x_diff)

            draw_text(img, f'x, y', FONT, (x_min, y_min), FONT_SCALE, BORDER_THICKNESS, WHITE, BLACK)
            if x_min < x_max:
                sec_box = np.array([[x_min, y_min], [x1, y1], [x2, y2], [x_max, y_max]])
                if y1 <= y2:
                    delta = np.array([-1*abs(x1 - x_min)//2, abs(y1 - y_min)//2])
                    delta = -1*delta if x_min > x1 else delta
                else:
                    delta = np.array([-1*abs(x1 - x_min)//2, -1*abs(y1 - y_min)//2])
                    delta = -1*delta if x_min > x1 else delta
            else:
                sec_box = np.array([[x_max, y_max], [x2, y2], [x1, y1], [x_min, y_min]])
                if y1 >= y2:
                    delta = np.array([-1*abs(x1 - x_min)//2, abs(y1 - y_min)//2])
                    delta = -1*delta if x_min > x1 else delta
                else:
                    delta = np.array([-1*abs(x1 - x_min)//2, -1*abs(y1 - y_min)//2])
                    delta = -1*delta if x_min > x1 else delta
            cv2.drawContours(img,[sec_box],0, YELLOW, BORDER_THICKNESS)
           

            sec_box = sec_box + delta

            self.counting_lines += [((x_min, y_min), (x_max, y_max))]
            self.line_counts += [0]
            self.boxes += [cv2.boundingRect(sec_box)]
            cv2.line(img, (x_min, y_min), (x_max, y_max), GREEN, BORDER_THICKNESS)
            # cv2.drawContours(img,[box],0, GREEN, BORDER_THICKNESS)
            cv2.drawContours(img,[sec_box],0, RED, BORDER_THICKNESS)

        img = cv2.resize(img, (int(width * VID_SCALE), int(height * VID_SCALE)))
        cv2.imshow('Output', img)
        cv2.waitKey(0) if stop else cv2.waitKey(1)
        cap_cpy.release()
        cv2.destroyAllWindows()



    def analyse_movement(self, area=DETECTION_AREA, duration=10):

        if self.boxes is None or len(self.boxes) == 0:
            self.boxes = []
            box_areas = set()
            duration = min(self.video_time.seconds, duration)
            for line_idx in range(len(self.counting_lines)):
                res_areas = self.__analyse_movement_by_line__(line_idx, area, duration)
                box_areas = box_areas.union(res_areas)

            cv2.destroyAllWindows()
            if not len(box_areas): return;
            med_ratio = statistics.median(list(box_areas))
            self.conf = max(min(med_ratio, 0.60), 0.40)
        self.boxes = np.array(self.boxes, np.uint32)
        X_min, Y_min, X_max, Y_max = 3000, 3000, 0, 0
        for (x_min, y_min, w, h) in self.boxes:
            x_max, y_max = x_min + w, y_min + h
            X_min, X_max = min(X_min, x_min), max(X_max, x_max)
            Y_min, Y_max = min(Y_min, y_min), max(Y_max, y_max)
        self.roi = np.array([X_min, Y_min, X_max - X_min, Y_max - Y_min], np.uint32)

    def process(self, goal=GOAL, conf=CONFIDENCE_THRESHOLD, nms=NMS_THRESHOLD, area=DETECTION_AREA,
        frame_dist=FRAME_TRACKING_DIST, verbose=False, display=False, car_info=False):

        self.goal = goal
        self.display = display

        curr_frame_count = 0
        ret, img = self.cap.read()
        height, width, _ = img.shape
        if conf != CONFIDENCE_THRESHOLD:
            self.conf = conf

        if not len(self.counting_lines):
            self.detect_lines()
            self.analyse_movement()

        if len(self.counting_lines) and not len(self.boxes):
            self.analyse_movement()

        while(ret):

            curr_frame_count += 1
            if not curr_frame_count % self.vid_speed: continue;

            img, curr_center_points = self.__extract_info_from_frame__(img, nms, frame_dist, car_info, curr_frame_count)

            if display:
                for idx, (box,(starting_point, ending_point)) in enumerate(zip(self.boxes, self.counting_lines)):
                    cv2.rectangle(img, box, BLUE, BORDER_THICKNESS)
                    draw_text(img, f'Box {idx}', FONT, (box[0], box[1]), FONT_SCALE, BORDER_THICKNESS, WHITE, BLUE)

                    cv2.line(img, starting_point, ending_point, GREEN, BORDER_THICKNESS)
                    cv2.putText(img, f'L{idx}', (starting_point[0], starting_point[1]-7), FONT, FONT_SCALE, GREEN, BORDER_THICKNESS, LINE_TYPE)

                    draw_text(img, f'L{idx} : {self.line_counts[idx]}', FONT, (0, (idx+2)*50), FONT_SCALE, BORDER_THICKNESS, WHITE, BLACK)
                draw_text(img, f'Car Count: {len(curr_center_points)}', FONT, (0, 50), FONT_SCALE, BORDER_THICKNESS, WHITE, BLACK)

                if self.roi is not None:
                    cv2.rectangle(img, self.roi, RED, BORDER_THICKNESS+1)
                    draw_text(img, f'ROI', FONT, (self.roi[0], self.roi[1]), FONT_SCALE, BORDER_THICKNESS, WHITE, RED)

                img = cv2.resize(img, (int(width * VID_SCALE), int(height * VID_SCALE)))
                cv2.imshow('Output', img)
            if cv2.waitKey(1) == 27: break;

            if verbose:
                curr_time = str(datetime.timedelta(seconds = curr_frame_count // self.fps))
                os.system('cls')
                print('Current Video Time    :', curr_time, curr_frame_count%self.fps)
                print('Total Video Time      :', self.video_time, self.fps)
                print('Confidence Threshold  :', self.conf)
                print('Total Car Count       :', self.id_counter)
                print('Current Car Count     :', len(curr_center_points))
                print('Total Cars Crossed    :', len(self.crossed_ids))
                print('Line Count            :', len(self.counting_lines))
                print('API Calls             :', self.api_calls)
                print('Car Count By Line     :', self.line_counts)
                print('Read Plates           :', [v for _,v in self.car_info.items()])

            self.vehicle_map = {k:v for k,v in self.vehicle_map.items() if v in curr_center_points}
            ret, img = self.cap.read()

        cv2.destroyAllWindows()
        self.cap.release()



    def get_statistics(self):
        return {
            'Total Car Count'       :   self.id_counter,
            'Total Cars Crossed'    :   len(self.crossed_ids),
            'Line Count'            :   len(self.counting_lines),
            'Line List'             :   self.counting_lines,
            'Car Count By Line'     :   self.line_counts,
            'Video FPS'             :   self.fps,
            'Video Duration'        :   self.video_time
        }

    def get_cars_info(self):
        return self.car_info.copy()



if __name__ == '__main__':
    video_path = r'C:\Zdaly\Zdaly-Vedio-Analysis\res\vids\british_highway_traffic.mp4'
    vd_obj = VehicleCounter(video_path)

    # print('Drawing Lines...')
    # vd_obj.detect_lines(duration=30, stop=True)
    # lines = draw_lines(video_path)
    # vd_obj = VehicleCounter(video_path, counting_lines=lines)

    # print('Processing Video...')
    # vd_obj.analyse_movement(duration = 10)

    vd_obj.process(verbose=True, display=True, car_info=True)
    # cars = vd_obj.get_cars_info()

    # vd_obj.df.to_csv('res.csv', index=False)