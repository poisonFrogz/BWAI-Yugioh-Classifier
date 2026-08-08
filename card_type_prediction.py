import resnet_model
import cv2
import torch
import datasets_type as datasets
import os

card_types = ['effect','fusion','link','normal','ritual','spell','synchro','trap','xyz']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = resnet_model.CnnModel(num_classes=9)
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

    card_type = card_types[predicted.item()]

    return card_type

rotated_path = 'C:/Users/Robyn/Desktop/Research Project/auto_rotate_cards/'

all_cards = os.listdir(rotated_path)
for card in all_cards:
    card_path = f'C:/Users/Robyn/Desktop/Research Project/auto_rotate_cards/{card}'
    loaded_card = cv2.imread(card_path)
    prediction = predict_type(loaded_card)

    print(f'Card: {card} ¦ Predicted: {prediction}')



