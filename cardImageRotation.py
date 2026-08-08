import os
import cv2
import torch

import orientation_model
import datasets_orientation as datasets

output_path = 'C:/Users/Robyn/Desktop/Research Project/auto_rotate_cards/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = orientation_model.CnnModel(num_classes=2)
path = 'C:/Users/Robyn/Desktop/Research Project/Saved Model/orientation_model.pth'

model.load_state_dict(torch.load(path))
model = model.to(device)
model.eval()

def card_prediction(card):

    card_to_predict = datasets.get_prediction_tensor(card)
    card_to_predict = card_to_predict.to(device)

    with torch.no_grad():
        output = model(card_to_predict)

        _, predicted = torch.max(output, 1)

    return predicted.item()

def card_rotation(card):
    orientation = card_prediction(card)
    original_card = cv2.imread(card)
    base_name = os.path.basename(card)
    card_name = os.path.splitext(base_name)[0]

    filename = f'{card_name}_Rotated.jpg'
    save_path = os.path.join(output_path, filename)


    if orientation == 0:
        rotated = cv2.rotate(original_card,cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(save_path, rotated)


    elif orientation == 1:
        rotated = cv2.rotate(original_card,cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(save_path, rotated)





card_path = 'C:/Users/Robyn/Desktop/Research Project/cropped_cards/'
cards = os.listdir(card_path)
for card in cards:
    path = f'{card_path}{card}'
    card_rotation(path)




