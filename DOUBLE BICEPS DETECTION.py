# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose





def calculate_angle(a, b, c):
    a = np.array(a) # FIRST POINT
    b = np.array(b) # SECOND POINT
    c = np.array(c) # THIRD POINT
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle




 

def draw_text_with_background(image, text, position, text_color, bg_color, font_scale=1.3, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    bottom_left = position
    top_right = (position[0] + text_size[0], position[1] - text_size[1] - 10)
    text_position = (bottom_left[0], bottom_left[1] - 5)

    cv2.rectangle(image, bottom_left, top_right, bg_color, cv2.FILLED)

    cv2.putText(image, text, text_position, font, font_scale, text_color, thickness)





def draw_angles_with_background(image, angle, text_pos):
    
    rectangle_color = (128, 128, 128)
    text_color = (255, 255, 255)
    
    text_size = cv2.getTextSize(str(angle), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(image, (text_pos[0], text_pos[1] - text_size[1] - 2),
                  (text_pos[0] + text_size[0], text_pos[1]), rectangle_color, -1)
    
    cv2.putText(image, str(angle), text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                text_color, 2, cv2.LINE_AA)





# cap = cv2.VideoCapture(0) # WEBCAM
cap2 = cv2.VideoCapture('cbum_doublebiceps_video.mkv')
# image = cv2.imread('test.png')

with mp_pose.Pose(min_detection_confidence=0.8, min_tracking_confidence=0.8) as pose:
    while cap2.isOpened():
        
        
        ret, frame = cap2.read()
                
        if not ret:
            break
    
    
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        results = pose.process(image_rgb)
        
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        
        
        
        try:
            landmarks = results.pose_landmarks.landmark
    
    
    
            # FIRST ANGLE (LEFT ARM)
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # SECOND ANGLE (RIGHT ARM)
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    
            # CALCULATING THE ANGLES
            left_angle = round(calculate_angle(left_shoulder, left_elbow, left_wrist), 2)
            right_angle = round(calculate_angle(right_shoulder, right_elbow, right_wrist), 2)
            
            left_shoulder_angle = round(calculate_angle(left_elbow, left_shoulder, right_shoulder), 2)
            right_shoulder_angle = round(calculate_angle(right_elbow, right_shoulder, left_shoulder), 2)
            
            
            # OBS: all in BGR
            if left_angle < 60 or right_angle < 60 or left_angle > 110 or right_angle > 110 or left_shoulder_angle < 140 or right_shoulder_angle < 140:
                draw_text_with_background(image, "WRONG POSTURE: CHECK THE ANGLES!", (20, 40), (0, 0, 255), (0, 0, 0))  
            else:
                draw_text_with_background(image, "RIGHT POSTURE! GOOD WORK!", (20, 40), (0, 255, 0), (0, 0, 0))
            
            
            ######################################################################                        
                        
            # left_curve_points = []
            # for i in range(0, 101):
            #     t = i / 100.0
            #     x = (1 - t) ** 2 * left_shoulder[0] + 2 * (1 - t) * t * left_elbow[0] + t ** 2 * left_wrist[0]
            #     y = (1 - t) ** 2 * left_shoulder[1] + 2 * (1 - t) * t * left_elbow[1] + t ** 2 * left_wrist[1]
            #     left_curve_points.append((int(x * image.shape[1]), int(y * image.shape[0])))
    
            # cv2.polylines(image, [np.array(left_curve_points)], isClosed=False, color=(0, 0, 255), thickness=3)
            
            #######################################################################
                
            
            text_pos_left = tuple(np.add(np.multiply(left_elbow, [image.shape[1], image.shape[0]]).astype(int), (-20, -20)))
            text_pos_right = tuple(np.add(np.multiply(right_elbow, [image.shape[1], image.shape[0]]).astype(int), (-20, -20)))
            text_pos_shoulder_left = tuple(np.add(np.multiply(left_shoulder, [image.shape[1], image.shape[0]]).astype(int), (-20, -20)))
            text_pos_shoulder_right = tuple(np.add(np.multiply(right_shoulder, [image.shape[1], image.shape[0]]).astype(int), (-20, -20)))
    
    
            draw_angles_with_background(image, right_angle, text_pos_right)
            draw_angles_with_background(image, left_angle, text_pos_left)
            draw_angles_with_background(image, left_shoulder_angle, text_pos_shoulder_left)
            draw_angles_with_background(image, right_shoulder_angle, text_pos_shoulder_right)
    
            
    
        except:
            pass
            
    
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=5),
                                  mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                                  )
    
        cv2.imshow('POSE DETECTION', image)
    
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # cv2.waitKey(0)
    cap2.release()
    cv2.destroyAllWindows()