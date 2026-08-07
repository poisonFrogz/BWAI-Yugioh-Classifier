import cv2
import numpy as np
import os



def crop_card(card, margin=30):
    card_BGR = cv2.imread(card)

    card_gray = cv2.cvtColor(card_BGR, cv2.COLOR_BGR2GRAY)

    _, card_threshold = cv2.threshold(card_gray, 170, 255, cv2.THRESH_BINARY_INV)

    contours, hierarchy = cv2.findContours(card_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        x,y,w,h = cv2.boundingRect(largest_contour)

        start_x = x + margin
        start_y = y + margin

        end_x = (x+w) - margin
        end_y = (y+h) - margin

        if start_x < end_x and start_y < end_y:

            cropped_card = card_BGR[start_y:end_y, start_x:end_x]

        else:
            print("Error, Margin too large")

    return cropped_card