
# Python program to implement 
# Webcam Motion Detector
  
# importing OpenCV, time and Pandas library
import cv2

BLUR_KERNEL_SIZE = 31
BLUR_KERNEL = (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE)

# Capturing video
video = cv2.VideoCapture(0)
# capture video with 30 fps
video.set(cv2.CAP_PROP_FPS, 60)
  
# Infinite while loop to treat stack of image as video
last_frame = None
while True:
    # Reading frame(image)
    check, frame = video.read()
    
    # Initializing motion = 0(no motion)
    motion = 0
  
    # Converting color image to gray_scale image
    # Converting gray scale image to GaussianBlur so that change can be find easily
    processed_frame = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), BLUR_KERNEL, 0)
  
    # In first iteration we assign the value of last_frame to our first frame
    if last_frame is None:
        last_frame = processed_frame
        continue
    
    diff_frame = cv2.absdiff(last_frame, processed_frame)
    last_frame = processed_frame
    # Difference between static background 
    # and current frame(which is GaussianBlur)
  
    # If change in between static background and
    # current frame is greater than 30 it will show white color(255)
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
  
    # Finding contour of moving object
    cnts,_ = cv2.findContours(thresh_frame.copy(), 
                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        motion = 1
  
        (x, y, w, h) = cv2.boundingRect(contour)
        # making green rectangle around the moving object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
  
  
    # Displaying image in gray_scale
    cv2.imshow("Gray Frame", processed_frame)
  
    # Displaying the difference in currentframe to
    # the staticframe(very first_frame)
    cv2.imshow("Difference Frame", diff_frame)
  
    # Displaying the black and white image in which if
    # intensity difference greater than 30 it will appear white
    cv2.imshow("Threshold Frame", thresh_frame)
  
    # Displaying color frame with contour of motion of object
    cv2.imshow("Color Frame", frame)
  
    key = cv2.waitKey(1)
    # if q entered whole process will stop
    if key == ord('q'):
        break
  
# Appending time of motion in DataFrame
# for i in range(0, len(time), 2):
#     df = df.append({"Start":time[i], "End":time[i + 1]}, ignore_index = True)
  
# # Creating a CSV file in which time of movements will be saved
# df.to_csv("Time_of_movements.csv")
  
video.release()
  
# Destroying all the windows
cv2.destroyAllWindows()