import cv2
import numpy as np

def process_frame(frame):

    blur = cv2.blur(frame, (50, 50))
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    lower_blue = np.array([50, 150, 50])
    upper_blue = np.array([140, 255, 255])
     
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = cv2.bitwise_or(mask_red, mask_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
    frame_status = ""
    max_contour = None
    second_contour = None
    max_area = 0

    second_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
    
        if area > max_area:
            second_area = max_area
            second_contour = max_contour
            max_area = area
            max_contour = contour
            
        elif area > second_area:
            second_area = area
            second_contour = contour
        
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        x2, y2, w2, h2 = cv2.boundingRect(second_contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
   
        if (x < x2 + w2 and x + w > x2 and y < y2 + h2 and y + h > y2 or max_area < 107000):   
            frame_status = "COLISAO DETECTADA"
        elif x + w < x2:  
            frame_status = "ULTRAPASSOU A BARREIRA"
       
    return frame, frame_status


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, frame_status = process_frame(frame)
        cv2.putText(processed_frame, frame_status, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Video Processing", processed_frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

process_video("q1/q1A.mp4")
