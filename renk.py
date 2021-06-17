import cv2
import numpy as np


# video ile görüntüyü işleme

# fileName='asd.mp4'
# cap = cv2.VideoCapture('asd.mp4')



# web kamerası aracılığıyla video çekme
cap = cv2.VideoCapture(0)


while (1):
    _, img = cap.read()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # kırmızı renk

    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)

    # mavi renk

    blue_lower = np.array([99, 115, 150], np.uint8)
    blue_upper = np.array([110, 255, 255], np.uint8)

    # ysarı renk

    yellow_lower = np.array([22, 60, 200], np.uint8)
    yellow_upper = np.array([60, 255, 255], np.uint8)

    # beyaz renk
    white_lower = np.array([0, 0, 200], np.uint8)
    white_upper = np.array([180, 20, 255], np.uint8)

    # siyah renk

    black_lower = np.array([0, 0, 0], np.uint8)
    black_upper = np.array([180, 255, 30], np.uint8)

    # tüm renkler birlikte

    red = cv2.inRange(hsv, red_lower, red_upper)
    blue = cv2.inRange(hsv, blue_lower, blue_upper)
    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    white = cv2.inRange(hsv, white_lower, white_upper)
    black = cv2.inRange(hsv, black_lower, black_upper)

    # Morfolojik Dönüşüm, Genişleme

    kernal = np.ones((5, 5), "uint8")

    red = cv2.dilate(red, kernal)
    res_red = cv2.bitwise_and(img, img, mask=red)

    blue = cv2.dilate(blue, kernal)
    res_blue = cv2.bitwise_and(img, img, mask=blue)

    yellow = cv2.dilate(yellow, kernal)
    res_yellow = cv2.bitwise_and(img, img, mask=yellow)

    white = cv2.dilate(white, kernal)
    res_white = cv2.bitwise_and(img, img, mask=white)

    black = cv2.dilate(black, kernal)
    res_black = cv2.bitwise_and(img, img, mask=black)

    # kırmızı takip
    (contours, hierarchy) = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "Kirmizi", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))

            # mavi takip
    (contours, hierarchy) = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Mavi", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))

    # sarı takip
    (contours, hierarchy) = cv2.findContours(yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Sari", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

    # beyaz takip
    (contours, hierarchy) = cv2.findContours(white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.putText(img, "Beyaz", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))

    # siyah takip
    (contours, hierarchy) = cv2.findContours(black, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv2.putText(img, "Siyah", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0))

    cv2.imshow("Renk Takip Yazilimi", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
