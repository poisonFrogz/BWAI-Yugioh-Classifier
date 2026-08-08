import cv2
import os
import numpy as np

def is_filled(card_bgr, x,y,w,h, white_threshold=110, max_white_ratio=0.80):
    box_crop = card_bgr[y:y+h, x:x+w]
    gray_crop = cv2.cvtColor(box_crop, cv2.COLOR_BGR2GRAY)

    total_pixels = gray_crop.size
    white_pixels = np.sum(gray_crop > white_threshold)
    white_ratio = white_pixels / total_pixels

    if white_ratio > max_white_ratio:
        return "empty"
    else:
        return "filled"


def crop_save_cards (board_state, cards_output, min_area=25000, max_area=200000):
    # Load Board, turn it gray
    board_bgr = cv2.imread(board_state)
    board_gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)

    #Threshold to make cards noticeable for contouring
    _, card_threshold = cv2.threshold(board_gray, 170, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(card_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found_cards = []
    saved_cards = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x,y,w,h = cv2.boundingRect(contour)

            aspect_ratio = float(w) / h

            is_portrait = (0.5 <= aspect_ratio <= 0.85)
            is_landscape = (1.15 <= aspect_ratio <= 2.0)

            if not (is_portrait or is_landscape):
                print("Ignored non card")
                continue

            status = is_filled(board_bgr, x,y,w,h)

            if status == 'filled':
                cropped_card = board_bgr[y:y + h, x:x + w]
                found_cards.append(cropped_card)

                filename = f'cropped_card_{saved_cards}.jpg'
                save_path = os.path.join(cards_output, filename)
                cv2.imwrite(save_path, cropped_card)

                print(f'{filename} saved to {cards_output}')
                saved_cards += 1

            else:
                print(f'Empty Zone')

        else:
            pass

    print(f'Cards extracted, {saved_cards} saved to {cards_output}')
    return found_cards

board_image_path = 'C:/Users/Robyn/Desktop/Research Project/board_states/board_state_1.jpg'
output_folder = 'C:/Users/Robyn/Desktop/Research Project/cropped_cards'

crop_save_cards(board_image_path, output_folder)