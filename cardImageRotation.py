import os
import cv2
import torch
import datasets_orientation as datasets

def card_prediction(card, device, model):

    card_to_predict = datasets.get_prediction_tensor(card)
    card_to_predict = card_to_predict.to(device)

    with torch.no_grad():
        output = model(card_to_predict)

        _, predicted = torch.max(output, 1)

    return predicted.item()

def card_rotation(card, device, model, output_path):
    original_card = cv2.imread(card)
    orientation = card_prediction(original_card, device, model)
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

    return save_path



