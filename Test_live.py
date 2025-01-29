"""
_________________________________________________________________________
                        

           Project : Computer vision project (Automate  Game)
_________________________________________________________________________
Students : *********/**********
_________________________________________________________________________
py version : 3.8
_________________________________________________________________________
Libraries :  cv2, numpy, time
, mss ,pygetwindow , pyautogui , pytesseract
_________________________________________________________________________
"""
                        #importing  modules
#you can install the listed packages by running:  '!pip install -r requirements.txt'
import pytesseract
import pyautogui
import pygetwindow as gw
import time
import cv2
import numpy as np
import mss
# _________________________________________________________________________________________
cx = 0
cy = 0
# define frame processing
MAX_FPS=5
MIN_FRAME_TIME=1/MAX_FPS
# define list for polyline test
detection_branch = []
detection_number = []
detection_energy =[]
# import pytesseract path file
pytesseract.pytesseract.tesseract_cmd =r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
number_1 ='1'
number_2 ='2'
number_3 ='3'
#img for Glass detection
img_match_Glass = cv2.imread('Glass_3.jpg')
# _________________________________________________________________________________________
#define Glass detection function
def detection_Glass(img_match_Glass):
    try:
        #crop bounding box in Root
        cropped_image = bigger[midpoint_y: cy_screen2 + int(cy_screen2 / 2), x1_1 - 10:x1_2 + 10]
        # resize
        cropped_image = cv2.resize(cropped_image, (300, 300))
        # convert to gray scale
        cropped_image_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        # Apply canny edage
        edged_cropped_image_gray = cv2.Canny(cropped_image_gray, 100, 255)
        #________________________________________________________________
        # resize
        cropped_image_match = cv2.resize(img_match_Glass, (300, 300))
        # convert to gray scale
        cropped_image_gray_match = cv2.cvtColor(cropped_image_match, cv2.COLOR_BGR2GRAY)
        # Apply canny edage
        edged_cropped_image_gray_match = cv2.Canny(cropped_image_gray_match, 100, 255)
        # ________________________________________________________________
        #finding contours for two images
        contours_crop, hierarchy_crop = cv2.findContours(edged_cropped_image_gray, 2, 1)
        cnt1 = contours_crop[0]
        contours_crop_match, hierarchy_crop_match = cv2.findContours(edged_cropped_image_gray_match, 2, 1)
        cnt2 = contours_crop_match[0]
        # Apply match shapes on two images
        ret = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
        #state for ret match
        if ret < 1.5:
            print("Glass detection and ret of match = ", ret)
        else:
            print("No Glass detectionand ret of match = ", ret)
    except cv2.error or TypeError or NameError or IndexError:
        print("none")
# _________________________________________________________________________________________
#define Hit counter for number detection
def counter_hit (counter,x_click ,y_click  ):
  print("Countdown begins")
  for i in reversed(range(1, counter)):
    print("count hit number = ",i)
    pyautogui( x= x_click , y = y_click )
    time.sleep(0.2)
  print("Time's up!,end hits")
# _________________________________________________________________________________________
#define function to calculate center of boudning box
def pega_centro(x, y, w, h):
        x1 = int(w / 2)

        y1 = int(h / 2)

        cx = x + x1

        cy = y + y1

        return cx, cy
# _________________________________________________________________________________________
#Applies a binary mask to the bottom half of an image
def apply_bottom_binary_mask(image, mask_size):
    """Applies a binary mask to the bottom half of an image while leaving the top half unchanged.
        after that will apply mask_player """
    height, width = image.shape[:2]
    # Create the binary mask
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[int(height // 1.42):int(height // 1.2), :] = 255  # Set bottom half to white (foreground)
    # Apply the mask using bitwise AND operation
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

#_________________________________________________________________________________________
while True:

        active_window = gw.getActiveWindow()
        #print(gw.getAllTitles())
        #  if state to check the game window
        if active_window and hasattr(active_window, 'title') and active_window.title.find("Play games")!= -1   :

            #print(f"Active window title: {active_window}")
            #print(gw.getAllTitles())
            #print(active_window.width)
            #print(type(active_window.width))
            img = None
            t0 = time.time()
            n_frames = 1
            #monitor = {"top": active_window.top, "left": active_window.left, "width": active_window.width, "height": active_window.height}
            with mss.mss() as sct:
                while True:

                    counter = 0
                    start = time.time()
                    monitor = {"top": active_window.top, "left": active_window.left, "width": active_window.width,
                               "height": active_window.height}
                    #img = sct.grab(monitor)
                    #img = np.array(img)  # Convert to NumPy array
                    bigger = sct.grab(monitor)
                    bigger = np.array(bigger)  # Convert to NumPy array

                    # define flags
                    active = False
                    active_2 = False
                    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR color
                    #print(monitor)
                    mask_size = bigger.shape[1]  # Full width of the image
                    gray_image = cv2.cvtColor(bigger, cv2.COLOR_BGR2GRAY)
                #_________________________________________________________________________________________
                                            # Apply THRESH Mask binary
                    ret, thresh1 = cv2.threshold(gray_image, 90, 255, cv2.THRESH_BINARY)
                    # get foreground
                    foreground = cv2.bitwise_and(bigger, bigger, mask=thresh1)
                    # get background
                    background = cv2.bitwise_not(thresh1)
                    #Apply binary mask on the top half of image
                    masked_image = apply_bottom_binary_mask(foreground, mask_size)
                #_________________________________________________________________________________________
                                                    # HSV color range

                    # Adjust these values based on your target Root of tree
                    lower_root_color = np.array([8, 147, 129])
                    upper_root_color = np.array([16, 220, 208])
                    # Adjust these values based on your target Screen left and right  (gray screen )
                    lower_screen_color = np.array([0, 0, 63])
                    upper_screen_color = np.array([0, 0, 66])
                    # Adjust these values based on your target player
                    lower_player_color = np.array([0, 112, 229])
                    upper_player_color = np.array([13, 138, 255])
                    # Adjust these values based on your target detection number
                    lower_number_color = np.array([20, 116, 219])
                    upper_number_color = np.array([29, 155, 255])
                    # Adjust these values based on your target detection energy
                    lower_green_energy = np.array([40, 40, 40])
                    upper_green_energy = np.array([80, 255, 255])
                #_________________________________________________________________________________________
                    # convert img from BGR to  HSV color space
                    hsv_image_foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV)
                    hsv_image_orginal = cv2.cvtColor(bigger, cv2.COLOR_BGR2HSV)
                    hsv_image_player = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)
                    # Create 4 HSV color mask for  root / screen / player / energy
                    mask_root_color = cv2.inRange(hsv_image_foreground, lower_root_color, upper_root_color)
                    mask_screen_color = cv2.inRange(hsv_image_orginal, lower_screen_color, upper_screen_color)
                    mask_player_color = cv2.inRange(hsv_image_player , lower_player_color, upper_player_color)
                    mask_energy_color = cv2.inRange(hsv_image_foreground, lower_green_energy, upper_green_energy)
                    # Create a 5x5 kernel of ones
                    kernel = np.ones((5, 5), np.uint8)
                    # Apply morphological operation on masks
                    closed_mask_root = cv2.morphologyEx(mask_root_color, cv2.MORPH_CLOSE, kernel)
                    closed_mask_screen = cv2.morphologyEx(mask_screen_color, cv2.MORPH_CLOSE, kernel)
                    closed_mask_player = cv2.morphologyEx(mask_player_color, cv2.MORPH_CLOSE, kernel)
                    closed_mask_energy = cv2.morphologyEx(mask_energy_color, cv2.MORPH_CLOSE, kernel)
                    # Apply dilation on masks
                    dilation_closed_mask_root = cv2.dilate(closed_mask_root, kernel, iterations=1)
                    dilation_closed_mask_energy = cv2.dilate(closed_mask_energy, kernel, iterations=1)
                #_________________________________________________________________________________________
                                                # Find Canny edges for masks
                    #  Canny edges for root
                    edged_root = cv2.Canny(dilation_closed_mask_root, 50, 150)
                    # cv2.imshow("dilation", edged_root)
                    #  Canny edges for screen
                    edged_screen = cv2.Canny(closed_mask_screen, 50, 200)
                    #  Canny edges for player
                    edged_player = cv2.Canny(closed_mask_player, 50, 200)
                    #  Canny edges for energy
                    edged_energy = cv2.Canny(dilation_closed_mask_energy, 50, 200)
                #_________________________________________________________________________________________
                                                # Finding Contours
                    #contours root
                    contours_root, hierarchy_1 = cv2.findContours(mask_root_color, cv2.RETR_EXTERNAL,
                                                                  cv2.CHAIN_APPROX_NONE)
                    # contours screen
                    contours_screen, hierarchy_2 = cv2.findContours(mask_screen_color, cv2.RETR_EXTERNAL,
                                                                    cv2.CHAIN_APPROX_NONE)
                    # contours player
                    contours_player, hierarchy_3 = cv2.findContours(mask_player_color, cv2.RETR_EXTERNAL,
                                                                    cv2.CHAIN_APPROX_NONE)
                    # contours energy
                    contours_energy, hierarchy_4 = cv2.findContours(dilation_closed_mask_energy, cv2.RETR_EXTERNAL,
                                                                    cv2.CHAIN_APPROX_NONE)
                    # cv2.drawContours(bigger, contours_root, -1, (0, 255, 0), 3)
                    # cv2.drawContours(bigger, contours_screen, -1, (0, 100, 100), 3)
                    # cv2.drawContours(bigger, contours_player, -1, (255, 255, 255), 3)
                    # cv2.drawContours(bigger, contours_energyr, -1, (255, 255, 255), 3)
                #_________________________________________________________________________________________
                                        # Finding Contour of energy
                            # sorted contours of enery by area using sorted function to get max area
                    try:
                        sorted_contours_by_area_energy = sorted(contours_energy, key=cv2.contourArea, reverse=True)
                        for contour in contours_energy:
                            # get dimensions [x,y,w,h] for the main energy box only
                            x_e, y_e, w_e, h_e = cv2.boundingRect( sorted_contours_by_area_energy[0])
                            # get moments  for main energy box (center)
                            M_number_energy = cv2.moments(sorted_contours_by_area_energy[0])
                            center_e = pega_centro(x_e, y_e, w_e, h_e)
                            detection_energy.append(center_e)
                            # draw the biggest contour
                            cv2.rectangle(bigger, (x_e, y_e),
                                          (x_e + w_e,
                                           y_e + h_e)
                                          , (255, 0, 255), 2)
                    except TypeError:
                        print('none')
                #_________________________________________________________________________________________
                                        # Finding Contours of Root
                                # sorted contours of Root by area using sorted function to get max area
                    try:
                        sorted_contours_by_area_root = sorted(contours_root, key=cv2.contourArea, reverse=True)
                    #_________________________________________________________________________________________
                                    # Finding Contour of the main Root
                        for contour in sorted_contours_by_area_root:
                        # get dimensions [x,y,w,h] for the main root box only
                            #x_root, y_root, w_root, h_root = cv2.boundingRect(sorted_contours_by_area_root[0])
                        # get moments  for main root box (center)
                            #M_root = cv2.moments(sorted_contours_by_area_root[0])
                            #if M_root['m00'] == 0:
                                #M_root['m00'] = 1
                            #else:
                                #cx_root = int(M_root['m10'] / M_root['m00'])
                                #cy_root = int(M_root['m01'] / M_root['m00'])
                            #new = int((cy_root + 30))
                        # drawing  circle on center of main root box
                            # cv2.circle(bigger, (x_root, cy_root), 5, (255, 255, 255), 1)

                        # drawing circle on right corner of the main root box in Blue
                            # cv2.circle(bigger, (x_root + w_root, y_root + h_root), 5, (0, 255, 0), 3)

                        # drawing circle on  left corner of the main root box in Blue
                            # cv2.circle(bigger, (x_root, y_root + h_root), 5, (255, 0, 0), 3)

                        # drawing  final rectangle on the main root
                            # cv2.rectangle(bigger, (x_root, y_root), (x_root + w_root, y_root + h_root), (0, 255, 255), 2)
                #_________________________________________________________________
                                    # Finding Contour of the brunch 1
                        # get dimensions [x,y,w,h] for the brunch 1 box only
                            x_b_1, y_b_1, w_b_1, h_b_1 = cv2.boundingRect(sorted_contours_by_area_root[1])
                        # get moments  for the brunch 1 box (center)
                            #M_b_1 = cv2.moments(sorted_contours_by_area_root[1])
                            #if M_b_1['m00'] == 0:
                                # M_b_1['m00'] = 1
                            #else:
                                # cx_b_1  = int(M_b_1['m10'] / M_b_1['m00'])
                                # cy_b_1 = int(M_b_1['m01'] / M_b_1['m00'])
                            center_b_1 = pega_centro(x_b_1, y_b_1, w_b_1, h_b_1)
                            detection_branch.append(center_b_1)
                        # drawing  final rectangle on the brunch 1
                            cv2.rectangle(bigger, (x_b_1, y_b_1), (x_b_1 + w_b_1, y_b_1 + h_b_1),
                                                  (100, 100, 100), 2)
                        # drawing  circle on center of the brunch 1 box
                            #cv2.circle(bigger, (center_b_1), 5, (255, 255, 255), 1)
                            # detection_branch.append(cx_b_1, cy_b_1)
                #_________________________________________________________________________________________
                                    # Finding Contour of the brunch 2
                        # get dimensions [x,y,w,h] for the brunch 2 box only
                            x_b_2, y_b_2, w_b_2, h_b_2 = cv2.boundingRect(sorted_contours_by_area_root[2])
                        # get moments  for the brunch 2 box (center)
                            #M_b_2 = cv2.moments(sorted_contours_by_area_root[2])
                            #if M_b_2['m00'] == 0:
                                # M_b_2['m00'] = 1
                            #else:
                                # cx_b_2  = int(M_b_2['m10'] / M_b_2['m00'])
                                # cy_b_2 = int(M_b_2['m01'] / M_b_2['m00'])
                            center_b_2 = pega_centro(x_b_2, y_b_2, w_b_2, h_b_2)
                            detection_branch.append(center_b_2)
                        # drawing  final rectangle on the brunch 2
                            cv2.rectangle(bigger, (x_b_2, y_b_2), (x_b_2 + w_b_2, y_b_2 + h_b_2), (100, 100, 100), 2)
                        # drawing  circle on center of the brunch 2 box
                            #cv2.circle(bigger, (center_b_2), 5, (255, 255, 255), 1)
                #_________________________________________________________________________________________
                                    # Finding Contour of the brunch 3
                        # get dimensions [x,y,w,h] for the brunch 3 box only
                            x_b_3, y_b_3, w_b_3, h_b_3 = cv2.boundingRect(sorted_contours_by_area_root[3])
                        # get moments  for the brunch 3 box (center)
                            #M_b_3 = cv2.moments(sorted_contours_by_area_root[3])
                            # if M_b_3['m00'] == 0:
                                # M_b_3['m00'] = 1
                            # else:
                                # cx_b_3  = int(M_b_3['m10'] / M_b_3['m00'])
                                # cy_b_3 = int(M_b_3['m01'] / M_b_3['m00'])
                            center_b_3 = pega_centro(x_b_3, y_b_3, w_b_3, h_b_3)
                            detection_branch.append(center_b_3)
                        # drawing  final rectangle on the brunch 3
                            cv2.rectangle(bigger, (x_b_3, y_b_3), (x_b_3 + w_b_3, y_b_3 + h_b_3), (100, 100, 100), 2)
                        # drawing  circle on center of the brunch 3 box
                            #cv2.circle(bigger, (center_b_3), 5, (255, 255, 255), 1)
                    except IndexError:
                        print("none")
                #_________________________________________________________________________________________
                             # sorted contours of screen by area using sorted function to get max area
                    try:
                        sorted_contours_by_area_screen = sorted(contours_screen, key=cv2.contourArea, reverse=True)
                #_________________________________________________________________________________________
                                        # Finding Contours of screen
                        for contour in contours_screen:
                                # Finding Contours of the left gray screen 1
                        # get dimensions [x,y,w,h] for the left gray screen 1   only
                            x_screen1, y_screen1, w_screen1, h_screen1 = cv2.boundingRect(
                                    sorted_contours_by_area_screen[0])
                        # get moments  for the left gray screen 1  (center)
                            M_screen1 = cv2.moments(sorted_contours_by_area_screen[0])
                            if M_screen1['m00'] == 0:
                                M_screen1['m00'] = 1
                            else:
                                cx_screen1 = int(M_screen1['m10'] / M_screen1['m00'])
                                cy_screen1 = int(M_screen1['m01'] / M_screen1['m00'])
                        # drawing  circle on center of the left gray screen 1
                                #cv2.circle(bigger, (cx_screen1, cy_screen1), 1, (255, 255, 255), 10)
                        # drawing  circle on corner of the left gray screen 1
                                # cv2.circle(bigger, (x_screen1 + w_screen1, y_screen1 + h_screen1), 5, (255, 255, 255), 3)
                                # right screen
                                #cv2.circle(bigger, (x_screen1, y_screen1 + h_screen1), 5, (20, 255, 255), 3)
                        # drawing  final rectangle on the left gray screen 1
                                #cv2.rectangle(bigger, (x_screen1, y_screen1), (x_screen1 + w_screen1, y_screen1 + h_screen1), (255, 255, 255), 2)
                #_________________________________________________________________________________________
                                    # Finding Contours of the right gray screen 2
                        # get dimensions [x,y,w,h] for the right gray screen 2  only
                            x_screen2, y_screen2, w_screen2, h_screen2 = cv2.boundingRect(
                                sorted_contours_by_area_screen[1])
                        # get moments  for the right gray screen 2  (center)
                            M_screen2 = cv2.moments(sorted_contours_by_area_screen[1])
                            if M_screen2['m00'] == 0:
                                M_screen2['m00'] = 1
                            else:
                                cx_screen2 = int(M_screen2['m10'] / M_screen2['m00'])
                                cy_screen2 = int(M_screen2['m01'] / M_screen2['m00'])
                        # drawing  circle on center of the right gray screen 2
                                #cv2.circle(bigger, (cx_screen2, cy_screen2), 1, (255, 255, 255), 10)
                        # drawing  circle on corner of the right gray screen 2
                                # cv2.circle(bigger, (x_screen2 + w_screen2, y_screen2 + h_screen2), 5, (255, 255, 255), 3)
                                # left screen
                                #cv2.circle(bigger, (x_screen2 + w_screen2, y_screen2 + h_screen2), 5, (255, 255, 255),3)
                        # drawing  final rectangle on the right gray screen 2
                                #cv2.rectangle(bigger, (x_screen2, y_screen2), (x_screen2 + w_screen2, y_screen2 + h_screen2), (255, 255, 255), 2)
                #_________________________________________________________________________________________
                                            # line between two centers
                        # drawing line between two center points of gray screen
                                #cv2.line(bigger, (cx_screen1, cy_screen1 + int(cy_screen1 / 5)), (cx_screen2, cy_screen2 + int(cy_screen2 / 5)),(255, 0, 0), 5)
                        try :
                        # get dimensions of start and end points of line
                            start_point_line_center_screen1 = (cx_screen1, cy_screen1 + int(cy_screen1 / 5))
                            end_point_line_center_screen2 = (cx_screen2, cy_screen2 + int(cy_screen2 / 5))
                        # Calculate the midpoint of line between two center points of gray screen
                            midpoint_x = int(
                                    (start_point_line_center_screen1[0] + end_point_line_center_screen2[0]) / 2)
                            midpoint_y = int(
                                    (start_point_line_center_screen1[1] + end_point_line_center_screen2[1]) / 2)
                        # Create a new point object for the midpoint of line
                            midpoint = (midpoint_x, midpoint_y)
                        # drawing  circle on middle  of  line
                                #cv2.circle(bigger, midpoint, 5, (0, 255, 0), -1)
                                #right (midpoint_x+int(midpoint_x/2), midpoint_y)
                                #left (midpoint_x-+int(midpoint_x/2), midpoint_y)
                #_________________________________________________________________________________________

                                                # right RIO detection BOX
                            pts = np.array([[midpoint_x, midpoint_y], [cx_screen1, midpoint_y],
                                        [cx_screen1, cy_screen1 + int(cy_screen1 / 4)],
                                        [midpoint_x, cy_screen1 + int(cy_screen1 / 4)]], np.int32)
                            cv2.polylines(bigger, [pts], True, (0, 0, 100), 2)
                                                # left RIO detection BOX
                            pts_2 = np.array([[midpoint_x, midpoint_y], [cx_screen2, midpoint_y],
                                                  [cx_screen2, cy_screen2 + int(cy_screen2 / 4)],
                                                  [midpoint_x, cy_screen2 + int(cy_screen2 / 4)]], np.int32)
                            cv2.polylines(bigger, [pts_2], True, (100, 0, 0), 2)
                                                # right RIO for energy detection BOX
                            pts_energy = np.array([[midpoint_x, int(midpoint_y / 1.2)], [cx_screen1, int(midpoint_y / 1.2)],
                                                   [cx_screen1, cy_screen1 + int(cy_screen1 / 4)],
                                                   [midpoint_x, cy_screen1 + int(cy_screen1 / 4)]], np.int32)
                            cv2.polylines(bigger, [pts_energy], True, (0, 0, 220), 2)
                                                # left RIO for energy detection BOX
                            pts_2_energy = np.array(
                                [[midpoint_x, int(midpoint_y / 1.2)], [cx_screen2, int(midpoint_y / 1.2)],
                                 [cx_screen2, cy_screen2 + int(cy_screen2 / 4)],
                                 [midpoint_x, cy_screen2 + int(cy_screen2 / 4)]], np.int32)
                            cv2.polylines(bigger, [pts_2_energy], True, (220, 0, 0), 2)
                        except NameError:
                            print("none")

                            '''for (x, y) in detection_branch:
                                        #  results of right RIO detection BOX and # right (midpoint_x+int(midpoint_x/2), midpoint_y)
                                        results = cv2.pointPolygonTest(np.array(pts, np.int32), (x, y), False)
                                        # results_2 of left RIO detection BOX and  # left (midpoint_x-int(midpoint_x/2), midpoint_y)
                                        results_2 = cv2.pointPolygonTest(np.array(pts_2, np.int32), (x, y), False)
                                        if results >= 0:
                                            cv2.circle(bigger, (x, y), 5, (255, 255, 255), 3)
                                            detection_branch.remove((x, y))
                                            #pyautogui.click(midpoint_x-int(midpoint_x/2), midpoint_y)
                                            #time.sleep(0.5)
                                            #x_click= midpoint_x-int(midpoint_x/2)
                                            #pyautogui.click(x=x_click, y=midpoint_y)
                                        elif results_2 >= 0:
                                            cv2.circle(bigger, (x, y), 5, (255, 255, 255), 3)
                                            detection_branch.remove((x, y))
                                            #pyautogui.click(midpoint_x+int(midpoint_x/2), midpoint_y)
                                            #time.sleep(0.5)
                                            #x_click = midpoint_x+int(midpoint_x/2)
                                            #pyautogui.click(x=x_click, y=midpoint_y)'''

                #_________________________________________________________________________________________
                                        #  ********************** Final boxes ipove player to detection brunchs **********************
                                # cv2.rectangle(bigger, (midpoint_x, midpoint_y), (cx_screen1, cy_screen1 + int(cy_screen1 / 4)), (50, 50, 50), 2)
                                # cv2.rectangle(bigger, (midpoint_x, midpoint_y), (cx_screen2, cy_screen2 + int(cy_screen2 / 2)), (50, 50, 50), 2)
                                # pts = np.array([[midpoint_x, midpoint_y], [cx_screen1, midpoint_y], [cx_screen1, cy_screen1 + int(cy_screen1 / 4)], [midpoint_x, cy_screen1 + int(cy_screen1 / 4)]], np.int32)
                                # cv2.polylines(bigger,[pts], True, (0, 255, 0), 2)
                    except IndexError or NameError:
                            print("none")
                #_________________________________________________________________________________________
                                    # Finding Contours of player
                    try:
                        for contour in contours_player:
                        # get max Area of counturs
                            c_player = max(contours_player, key=cv2.contourArea)
                        # get moments  for the player  (center)
                            M_player = cv2.moments(c_player)
                            if M_player['m00'] == 0:
                                M_player['m00'] = 1
                            else:
                                cx_player = int(M_player['m10'] / M_player['m00'])
                                cy_player = int(M_player['m01'] / M_player['m00'])
                                # get dimensions [x,y,w,h] for player only
                                x_player, y_player, w_player, h_player = cv2.boundingRect(c_player)
                                # drawing  circle on center of player
                                cv2.circle(bigger, (cx_player, cy_player), 1, (255, 255, 255), 1)
                                # drawing  final rectangle on face player
                                cv2.rectangle(masked_image , (x_player, y_player), (x_player + w_player, y_player + h_player),(100, 100, 255), 2)
                    except IndexError:
                        print("none")
                    try:
                        lines = cv2.HoughLines(edged_root, 1, np.pi / 180, 120)
                        for rho_1, theta_1 in lines[0]:
                            a_1 = np.cos(theta_1)
                            b_1 = np.sin(theta_1)
                            x0_1 = a_1 * rho_1
                            y0_1 = b_1 * rho_1
                            x1_1 = int(x0_1 + 1000 * (-b_1))
                            y1_1 = int(y0_1 + 1000 * (a_1))
                            x2_1 = int(x0_1 - 1000 * (-b_1))
                            y2_1 = int(y0_1 - 1000 * (a_1))
                            #cv2.line(bigger, (x1_1, y1_1), (x2_1, y2_1), (0, 0, 255), 1)
                            # cv2.circle(bigger, (x1_1, int(y1_1/2)), 5, (0, 0, 255), 3)
                        for rho_2, theta_2 in lines[1]:
                            a_2 = np.cos(theta_2)
                            b_2 = np.sin(theta_2)
                            x0_2 = a_2 * rho_2
                            y0_2 = b_2 * rho_2
                            x1_2 = int(x0_2 + 1000 * (-b_2))
                            y1_2 = int(y0_2 + 1000 * (a_2))
                            x2_2 = int(x0_2 - 1000 * (-b_2))
                            y2_2 = int(y0_2 - 1000 * (a_2))
                            #cv2.line(bigger, (x1_2, y1_2), (x2_2, y2_2), (0, 0, 255), 1)
                            # cv2.circle(bigger, (x1_2, int(y1_2 / 2)), 5, (0, 0, 255), 3)
                            # cv2.rectangle(bigger, (x1_2, midpoint_y), (x1_1, cy_screen2 + int(cy_screen2 / 2.5)), (100, 50, 50), 1)
                            # cv2.circle(bigger, (x1_2, midpoint_y), 5, (0, 0, 255), 3)
                            # cv2.circle(bigger, (x1_1, cy_screen2 + int(cy_screen2 / 2.5)), 5, (0, 0, 255), 3)
                            # cv2.line(bigger, (x1_2, midpoint_y), (x1_1, cy_screen2 + int(cy_screen2 / 2.5)), (0, 0, 255), 1)
                            # cv2.rectangle(bigger, (x1_2, y1_2), (x1_1, cy_screen2 + int(cy_screen2 / 2.5)), (2, 0, 50), 1)
                    except IndexError or TypeError or cv2.error or NameError:
                        print("none")
                #_________________________________________________________________________________________
                    try:
                        pts_root = np.array([[x1_2, midpoint_y], [x1_2, cy_screen2 + int(cy_screen2 / 2.5)],
                                             [x1_1, cy_screen2 + int(cy_screen2 / 2.5)], [x1_1, midpoint_y]], np.int32)
                        cv2.polylines(bigger, [pts_root], True, (0, 255, 0), 2)
                        mask_number_color = cv2.inRange(hsv_image_player, lower_number_color, upper_number_color)
                        closed_mask_color = cv2.morphologyEx(mask_number_color, cv2.MORPH_CLOSE, kernel)
                        # Apply dilation on mask
                        dilation_closed_mask_color = cv2.dilate(closed_mask_color, kernel, iterations=1)
                        edged_number = cv2.Canny(dilation_closed_mask_color, 100, 200)
                        contours_number, hierarchy = cv2.findContours(edged_number, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        sorted_contours_by_area_number_root = sorted(contours_number, key=cv2.contourArea, reverse=True)
                        try :
                            x_b, y_b, w_b, h_b = cv2.boundingRect(sorted_contours_by_area_number_root[0])
                            cv2.rectangle(bigger, (x_b, y_b), (x_b + w_b, y_b + h_b), (100, 100, 100), 2)
                            center_number = pega_centro(x_b, y_b, w_b, h_b)
                            detection_number.append(center_number)
                        except IndexError or NameError:
                            print('no contours for detection number')

                    except NameError :
                        print("none")
                #__________________________________________________________________________
                    try:
                        detection_branch = list(dict.fromkeys(detection_branch))
                        for (x, y) in detection_branch:
                                    #  results of right RIO detection BOX and # right (midpoint_x+int(midpoint_x/2), midpoint_y)
                                results = cv2.pointPolygonTest(np.array(pts, np.int32), (x, y), False)
                                    # results_2 of left RIO detection BOX and  # left (midpoint_x-int(midpoint_x/2), midpoint_y)
                                results_2 = cv2.pointPolygonTest(np.array(pts_2, np.int32), (x, y), False)
                                detection_branch.remove((x, y))
                                if results >= 0 :
                                        #print('detect right')
                                        #cv2.circle(bigger, (x, y), 5, (255, 255, 255), 3)
                                        #detection_branch.remove((x, y))

                                        #time.sleep(1)
                                        #pyautogui.click(midpoint_x-int(midpoint_x/2), midpoint_y)

                                        #break
                                        #x_click= midpoint_x-int(midpoint_x/2)
                                        #pyautogui.click(x=x_click, y=midpoint_y)
                                        #detection_branch = []
                                        #time.sleep(2)
                                        #_________________________________________________
                                        # results = -1
                                        print('detect right')
                                        #cv2.circle(bigger, (x, y), 5, (255, 255, 255), 3)
                                        # detection_branch.remove((x, y))
                                        # time.sleep(1)
                                        # pyautogui.click(midpoint_x - int(midpoint_x / 2), midpoint_y)
                                        print("active = ", active)
                                        active = True
                                        active_2 = True
                                        print("active = ", active)
                                        # time.sleep(1)
                                        'reverse to left BOX (left point) '
                                        x_click = midpoint_x - int(midpoint_x / 2)
                                        y_click = midpoint_y
                                        time.sleep(0.1)
                                        break
                                        # x_click= midpoint_x-int(midpoint_x/2)
                                        # pyautogui.click(x=x_click, y=midpoint_y)
                                        # detection_branch = []
                                        # time.sleep(2)
                                elif results_2 >= 0  :

                                        #print('detect left')

                                        #detection_branch = []
                                        #cv2.circle(bigger, (x, y), 5, (255, 255, 255), 3)
                                        #detection_branch.remove((x, y))

                                        #time.sleep(1)
                                        #pyautogui.click(midpoint_x+int(midpoint_x/2), midpoint_y)

                                        #break
                                        #detection_branch = []
                                        #time.sleep(2)
                                        # x_click = midpoint_x+int(midpoint_x/2)
                                        # pyautogui.click(x=x_click, y=midpoint_y)
                                        #___________________________________
                                        print('detect left')
                                        print("active = ", active)
                                        # detection_branch = []
                                        #cv2.circle(bigger, (x, y), 5, (255, 255, 255), 3)
                                        # detection_branch.remove((x, y))
                                        active = True
                                        active_2 = True
                                        print("active = ", active)
                                        # time.sleep(1)
                                        'reverse to Right BOX (Right point) '
                                        x_click = midpoint_x + int(midpoint_x / 2)
                                        y_click = midpoint_y
                                        time.sleep(0.1)
                                        # pyautogui.click(midpoint_x + int(midpoint_x / 2), midpoint_y)

                                        break
                                        # detection_branch = []
                                        # time.sleep(2)
                                        # x_click = midpoint_x+int(midpoint_x/2)
                                        # pyautogui.click(x=x_click, y=midpoint_y)
                                else :
                                        #print('detect nothing ')
                                        #detection_branch.remove((x, y))
                                        #detection_branch = []
                                        #time.sleep(2)
                                        #___________________________________________
                                        print('detect nothing ')
                                        active = True
                                        x_click = x_player
                                        y_click = y_player
                                        time.sleep(0.1)

                        #cv2.imshow("cropped_mask",cropped_image)
                        print("ok")
                        # cv2.imshow('cropped_image', cropped_image)
                    except  NameError or cv2.error:
                        print("none")
                    try :
                        detection_energy = list(dict.fromkeys(detection_energy))

                        for (x, y) in detection_energy:
                            # right
                            results_energy_1 = cv2.pointPolygonTest(np.array(pts_energy, np.int32), (x, y), False)
                            # left
                            results_energy_2 = cv2.pointPolygonTest(np.array(pts_2_energy, np.int32), (x, y), False)
                            detection_energy.remove((x, y))
                            if results_energy_1 >= 0:
                                cv2.circle(bigger, (center_e), 5, (0, 255, 0), 3)
                                print("green energy  detection!!!!")
                                active = True
                                x_click = midpoint_x + int(midpoint_x / 2)
                                y_click = midpoint_y
                                time.sleep(0.1)
                                break
                            elif results_energy_2 >= 0:
                                cv2.circle(bigger, (center_e), 5, (0, 255, 0), 3)
                                active = True
                                print("green energy  detection!!!!")
                                x_click = midpoint_x - int(midpoint_x / 2)
                                y_click = midpoint_y
                                time.sleep(0.1)
                                break
                            else:
                                print("green energy not detection")
                                time.sleep(0.1)

                    except IndexError:
                            print('none')
                    try :
                        detection_number = list(dict.fromkeys(detection_number))
                        for (x, y) in detection_number:

                            results_number = cv2.pointPolygonTest(np.array(pts_root, np.int32), (x, y), False)
                            detection_number.remove((x, y))
                            if results_number >= 0:
                                results_number = -1
                                cropped_image_number = masked_image[y_b: y_b + h_b, x_b:x_b + w_b]
                                cropped_image_number = cv2.resize(cropped_image_number, (300, 300))
                                hsv_image_cropped_2 = cv2.cvtColor(cropped_image_number, cv2.COLOR_BGR2HSV)
                                mask_number_color_2 = cv2.inRange(hsv_image_cropped_2, lower_number_color,
                                                                  upper_number_color)
                                closed_mask_color_2 = cv2.morphologyEx(mask_number_color_2, cv2.MORPH_CLOSE, kernel)
                                # Apply dilation on mask
                                dilation_closed_mask_color_2 = cv2.dilate(closed_mask_color_2, kernel, iterations=1)
                                edged_number_2 = cv2.Canny(dilation_closed_mask_color_2, 100, 200)
                                gray_1_2 = cv2.cvtColor(cropped_image_number, cv2.COLOR_BGR2GRAY)
                                text = pytesseract.image_to_string(mask_number_color)
                                print(text)
                                gray = np.float32(edged_number_2)
                                dst = cv2.cornerHarris(gray, 2, 1, 0.04)
                                num_corners = np.sum(dst > 0.01 * dst.max())
                                print(num_corners)
                                if (2700 > num_corners > 1500 ) or  (number_2 in text):
                                    print(" number 2 ")
                                    # flag if detection branch and number at the same time / ignore  function counter_hit() / and still update last x_click and y_click
                                    if active_2 != True :
                                        counter_hit(counter = 3)
                                        active == False
                                        break
                                    else :
                                        break
                                elif (1500 > num_corners > 900) or (number_1 in text):
                                    print(" number 1 ")
                                    # flag if detection branch and number at the same time / ignore  function counter_hit() / and still update last x_click and y_click
                                    if active_2 != True:
                                        counter_hit(counter=2)
                                        active == False
                                        break
                                    else :
                                        break
                                elif (2800 < num_corners) or (number_3 in text):
                                    print(" number 3 ")
                                    # flag if detection branch and number at the same time / ignore  function counter_hit() / and still update last x_click and y_click
                                    if active_2 != True:
                                        counter_hit(counter=4)
                                        active == False
                                        break
                                    else :
                                        break
                                else:
                                    print(" number 0 ")
                                    break

                    except  NameError or cv2.error:
                        print("none")
                    print("out fo loop")
                    detection_Glass(img_match_Glass)
                    cv2.imshow('Contours', bigger)
                    try:
                        if active == True :
                            pyautogui.click(x = x_click ,  y = y_click)
                            active = False
                            print("active = ",active)
                        time.sleep(0.3)
                    except NameError :
                        print("none")
                    #cv2.imshow('Contours', bigger)
                    time.sleep(max(0, MIN_FRAME_TIME - (time.time() - start)))
                    # Break loop and end test
                    key = cv2.waitKey(1)
                    if key == ord('q') :
                        break

                    #elapsed_time = time.time() - t0
                    #avg_fps = (n_frames / elapsed_time)
                    #print("Average FPS: " + str(avg_fps))
                    #n_frames += 1
            #time.sleep(1)
        else:
            print("No active window found or no title attribute")



