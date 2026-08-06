import os
import cv2
import torch
import cnnmodel
import datasets_orientation as datasets

def card_prediction(card):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = cnnmodel.CnnModel(num_classes=2)
    path = 'C:/Users/Robyn/Desktop/Research Project/Saved Model/orientation_model.pth'

    model.load_state_dict(torch.load(path))
    model = model.to(device)
    model.eval()

    card_to_predict = datasets.get_prediction_tensor(card)
    card_to_predict = card_to_predict.to(device)

    with torch.no_grad():
        output = model(card_to_predict)

        _, predicted = torch.max(output, 1)

    return predicted.item()

def card_rotation(card):
    orientation = card_prediction(card)
    original_card = cv2.imread(card)
    card_name = os.path.splitext(card)[0]



    if orientation == 0:
        rotated = cv2.rotate(original_card,cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(f'{card_name}Rotated.jpg',rotated)


    elif orientation == 1:
        rotated = cv2.rotate(original_card,cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(f'{card_name}Rotated.jpg', rotated)





#card_path = 'C:/Users/Robyn/Desktop/Research Project/YuGiOh board/'
#cards = os.listdir(card_path)
#for card in cards:
    #path = f'{card_path}{card}'
    #card_rotation(path)




