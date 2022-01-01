import cv2 as cv
import mediapipe as mp
import numpy as np
import random
import time

mp_drawing = mp.solutions.mediapipe.python.solutions.drawing_utils
mp_pose = mp.solutions.mediapipe.python.solutions.pose

# Video reading
cap = cv.VideoCapture('Videos/jump3.mp4')
# cap = cv.VideoCapture(0)

counter = 0
stage = None
start = 0
isSpawn = False
haveMon = False
monster_Arr = []
total_time = None
clear_time = 0
reward_rank = 'playing'

# Level modifier
# level = 'beginner'
level = 'intermediate'
# level = 'advance'

def calculate_angle(first, mid, end):
    a = np.array(first)
    b = np.array(mid)
    c = np.array(end)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    
    angle = np.abs(radians*180/np.pi)

    if angle > 180:
        angle = 360-angle
            
    return angle

def detect_jump(leg_status, hand_status):
    if leg_status and hand_status:
        res = True
    else:
        res = False
    return res

def detect_stand(aL, aR, bL, bR, cL, cR, dL, dR):
    if aL>150 and aR>150 and bL<10 and bR<10 and cL>165 and cR>165 and dL>165 and dR>165:
        res = True
    else:
        res = False
    return res

def leg_gesture(cL, cR, dL, dR):
    if cL<165 and cR<165 and dL>130 and dR>130:
        res = True 
    else:
        res = False
    return res

def hand_gesture(aL, aR, bL, bR):
    if aL>130 and aR>130 and bL>130 and bR>130:
        res = True
    else:
        res = False
    return res

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def monster_spawn(elapsed_time):
    if elapsed_time == 3:
        return True
    else:
        return False

def hp_mon(level):
    if level == 'beginner':
        return 1
    elif level == 'intermediate':
        return 2
    elif level == 'Advance':
        return 3

def init_monster(level):
    if level == 'beginner':
        return 5
    elif level == 'intermediate':
        return 10
    elif level == 'Advance':
        return 15

total_monster = init_monster(level)
current_monster = total_monster

def attack(monsters):
    monsters[0]['hp'] -= 1
    if monsters[0]['hp'] == 0:
        monsters.pop(0)
        global current_monster 
        current_monster -= 1

def init_timer(level):
    global total_time
    if level == 'beginner':
        total_time = 50
    elif level == 'intermediate':
        total_time = 100
    elif level == 'Advance':
        total_time = 150

def isGameover(current_time):
    global total_time
    global current_monster
    if current_monster == 0 or total_time == current_time:
        rewarding()
        return True
    else:
        return False

def rewarding():
    global total_monster
    global current_monster
    global reward_rank
    if current_monster == 0:
        reward_rank = 'Gold'
    elif (total_monster - int(total_monster * 0.7)) >= current_monster:
        reward_rank = 'Siver'
    elif (total_monster - int(total_monster * 0.5)) >= current_monster:
        reward_rank = 'Bronze'
    else:
        reward_rank = 'Fail'
        
init_timer(level)

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Rescale frame
        frame = rescaleFrame(frame, 0.5)

        # Recolor image
        img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img.flags.writeable = False

        # Make detection
        result = pose.process(img)

        # Recolor back to BGR
        img.flags.writeable = True
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

        # Extract Landmarks
        try:
            landmarks = result.pose_landmarks.landmark

            # Frame detection
            # isPlay = True
            # for index in range(11, 27):
            #     if index >= 17 and index <= 22:
            #         continue

            #     cx, cy = int(landmarks[index].x * img.shape[1]), int(landmarks[index].y * img.shape[0])
                
            #     if cx <= 160 or cx >= 480 or cy <= 0 or cy >= 480:
            #         # print(cy)
            #         # print(index)
            #         isPlay = False
            #         # err = 'Frame'
            # # if isPlay: 
            # #     err = None 

            # Get coordinates
            wristL = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            wristR = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            elbowL = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            elbowR = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

            shoulderL = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            shoulderR = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            hipL = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            hipR = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            kneeL = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            kneeR = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            ankleL = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            ankleR = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # Angle calculation
            elbowAngleL = calculate_angle(shoulderL, elbowL, wristL)
            elbowAngleR = calculate_angle(shoulderR, elbowR, wristR)

            shoulderAngleL = calculate_angle(hipL, shoulderL, elbowL)
            shoulderAngleR = calculate_angle(hipR, shoulderR, elbowR)

            hipAngleL = calculate_angle(shoulderL, hipL, kneeL)
            hipAngleR = calculate_angle(shoulderR, hipR, kneeR)

            kneeAngleL = calculate_angle(hipL, kneeL, ankleL)
            kneeAngleR = calculate_angle(hipR, kneeR, ankleR)
            
            # Leg_gesture    
            status = leg_gesture(hipAngleL, hipAngleR, kneeAngleL, kneeAngleR)

            # Hand Gesture     
            status2 = hand_gesture(elbowAngleL, elbowAngleR, shoulderAngleL, shoulderAngleR)

            # Gesture detection
            isJump = detect_jump(status, status2)
            isStand = detect_stand(elbowAngleL, elbowAngleR, shoulderAngleL, shoulderAngleR, hipAngleL, hipAngleR, kneeAngleL, kneeAngleR) 

            timer = int(time.perf_counter())
            elapsed_time = int(timer - start)

            isSpawn = monster_spawn(elapsed_time)

            game_status = isGameover(timer)

            if game_status:
                clear_time = int(time.perf_counter())

            if isSpawn and not game_status:
                isSpawn = False
                start = timer
                spawn_time = timer
                x = random.randrange(0, img.shape[1])
                y = random.randrange(0, img.shape[0])
                hp = hp_mon(level)
                monster = {
                    'spawn_time': spawn_time,
                    'x': x,
                    'y': y,
                    'hp': hp,
                }
                monster_Arr.append(monster)

            if isJump:
                stage = 'up'
            elif isStand and stage == 'up':
                stage = 'down'
                counter += 1
                attack(monster_Arr)

            # Monster display
            for index in range(0, len(monster_Arr)):
                center_pos = (monster_Arr[index]['x'], monster_Arr[index]['y'])
                text_font = cv.FONT_HERSHEY_SIMPLEX
                text_scale = 1
                text_thickness = 2
                text = str(monster_Arr[index]['hp'])
                text_size = cv.getTextSize(text, text_font, text_scale, text_thickness)
                text_pos = (int(center_pos[0] - text_size[0][0] / 2), int(center_pos[1] + text_size[0][1] // 2))
                
                cv.circle(img, center_pos, 40, (0, 0, 255), thickness=-3)
                cv.putText(img, text, text_pos, text_font, text_scale, (255,255,255), text_thickness, cv.LINE_AA)

            # Output detail
            cv.putText(img, 'Reps: ' + str(counter), (130,40), cv.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv.LINE_AA)
            cv.putText(img, 'Stage: ' + str(stage), (20,80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv.LINE_AA) 
            cv.putText(img, 'Leg: ' + str(status), (20,120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv.LINE_AA)
            cv.putText(img, 'Hand: ' + str(status2), (20,160), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv.LINE_AA)  
            cv.putText(img, 'Timer: ' + str(int(timer)) + 's', (20,200), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv.LINE_AA)
            cv.putText(img, 'Clear time: ' + str(int(clear_time)) + 's', (20,240), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv.LINE_AA)
            cv.putText(img, 'Reward rank: ' + str(reward_rank), (20,280), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv.LINE_AA)

            # Output Angle
            cv.putText(img, str(int(hipAngleL)),
                        tuple(np.multiply(hipL, [img.shape[1], img.shape[0]]).astype(int)),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(img, str(int(hipAngleR)),
                        tuple(np.multiply(hipR, [img.shape[1], img.shape[0]]).astype(int)),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

            cv.putText(img, str(int(kneeAngleL)),
                        tuple(np.multiply(kneeL, [img.shape[1], img.shape[0]]).astype(int)),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(img, str(int(kneeAngleR)),
                        tuple(np.multiply(kneeR, [img.shape[1], img.shape[0]]).astype(int)),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

            cv.putText(img, str(int(shoulderAngleL)),
                        tuple(np.multiply(shoulderL, [img.shape[1], img.shape[0]]).astype(int)),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(img, str(int(shoulderAngleR)),
                        tuple(np.multiply(shoulderR, [img.shape[1], img.shape[0]]).astype(int)),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        except:
            pass

        # Render detection
        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        # Show video
        cv.imshow('Video', img)

        # Exit
        if cv.waitKey(20) & 0xFF == ord('d'):
            break

    cap.release()
    cv.destroyAllWindows()