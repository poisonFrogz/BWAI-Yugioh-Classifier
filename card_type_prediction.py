import cnnmodel
import cv2
import torch
import datasets_type as datasets
import crop_card
import os

card_types = ['effect','fusion','link','normal','ritual','spell','synchro','trap','xyz']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = cnnmodel.CnnModel(num_classes=9)
path = 'C:/Users/Robyn/Desktop/Research Project/Saved Model/card_type_model.pth'

model.load_state_dict(torch.load(path))
model = model.to(device)
model.eval()

def predict_type(card):


    card_type_to_predict = datasets.get_prediction_tensor(card)
    card_type_to_predict = card_type_to_predict.to(device)

    with torch.no_grad():
        output = model(card_type_to_predict)

        _, predicted = torch.max(output, 1)

    for type in range(len(card_types)):
        if predicted == type:
            card_type = card_types[type]
            break

    return card_type

path = 'C:/Users/Robyn/Desktop/Research Project/rotated_cards/'

all_cards = os.listdir(path)
for card in all_cards:
    card_path = f'C:/Users/Robyn/Desktop/Research Project/rotated_cards/{card}'
    cropped_card = crop_card.crop_card(card_path)
    prediction = predict_type(cropped_card)
    print(f'Card:{card} Predicted:{prediction}')
    cv2.imshow('cropped_card',cropped_card)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



