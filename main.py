
import orientation_model
import resnet_model
import crop_card as cc
import cardImageRotation as cir
import card_type_prediction as ctp
import cardTextExtraction as cte
import string_match as sm
import torch
import easyocr
import os
import cv2
import itertools

card_types = ['effect','fusion','link','normal','ritual','spell','synchro','trap','xyz']

om_path = 'utils/models/orientation_model.pth'
rm_path = 'utils/models/card_type_model.pth'
card_text_path = 'utils/card_types/'
board_state_path = 'utils/board_states/'
board_state_output = 'utils/cropped_cards/'
cards_processed = 'utils/auto_rotate_cards/'
annotated_boards = 'utils/annotated_boards/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
om = orientation_model.CnnModel(num_classes=2)
rm = resnet_model.CnnModel(num_classes=9)
reader = easyocr.Reader(['en'], gpu=True)

om.load_state_dict(torch.load(om_path))
om = om.to(device)
om.eval()

rm.load_state_dict(torch.load(rm_path))
rm = rm.to(device)
rm.eval()

saved_cards = 0
board_states = os.listdir(board_state_path)

# for each state in board_states
for state in board_states:
    frame_types = []
    cards_text = []
    formatted_text = []
    # crop the board into cards
    boards_to_crop = os.path.join(board_state_path, state)
    saved_cards, current_board_cards = cc.crop_board(boards_to_crop, saved_cards, board_state_output)



    # rotate the cards
    current_processed_cards = []
    for card in current_board_cards:
        rotate_path = os.path.join(board_state_output, card)

        processed_path = cir.card_rotation(rotate_path, device, om, cards_processed)

        current_processed_cards.append(processed_path)

    # predict the card types

    print(current_processed_cards)
    for card_prediction in current_processed_cards:
        frame_type = ctp.predict_type(card_prediction, rm, device, card_types)
        frame_types.append(frame_type)
    print(frame_types)

    # predict card text
    for card_prediction in current_processed_cards:
        card_text = cte.extract_text(card_prediction, reader)
        cards_text.append(card_text)
    print(cards_text)

    # check card text, format to how it is actually displayed
    for text, card_type in zip(cards_text, frame_types):

        clean_text = sm.find_string(card_text_path, text, card_type)
        # store reformatted text in a list
        formatted_text.append(clean_text)
    print(formatted_text)
    # draw bounding boxes around cards on board, put card name on top
    cc.draw_boxes(boards_to_crop, formatted_text, annotated_boards)





