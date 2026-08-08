import torch
import datasets_type as datasets


def predict_type(card, model, device, card_types):


    card_type_to_predict = datasets.get_prediction_tensor(card)
    card_type_to_predict = card_type_to_predict.to(device)

    with torch.no_grad():
        output = model(card_type_to_predict)

        _, predicted = torch.max(output, 1)

    card_type = card_types[predicted.item()]

    return card_type







