import orientation_model
import resnet_model
import crop_card
import cardImageRotation as cir
import card_type_prediction as ctp
#import cardTextExtraction as cte
import torch
import easyocr
import os
import cv2

card_types = ['effect','fusion','link','normal','ritual','spell','synchro','trap','xyz']

om_path = 'C:/Users/Robyn/Desktop/Research Project/Saved Model/orientation_model.pth'
rm_path = 'C:/Users/Robyn/Desktop/Research Project/Saved Model/card_type_model.pth'
board_state_path = 'C:/Users/Robyn/Desktop/Research Project/board_states/'
board_state_output = 'C:/Users/Robyn/Desktop/Research Project/cropped_cards/'
cards_to_predict = 'C:/Users/Robyn/Desktop/Research Project/auto_rotate_cards/'

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
for state in board_states:
    boards_to_crop = os.path.join(board_state_path, state)
    saved_cards = crop_card.crop_board(boards_to_crop,saved_cards,board_state_output)
    saved_cards = saved_cards


cards_to_rotate = os.listdir(board_state_output)
for card in cards_to_rotate:
    rotate_path = os.path.join(board_state_output,card)
    cir.card_rotation(rotate_path,device,om,cards_to_predict)




