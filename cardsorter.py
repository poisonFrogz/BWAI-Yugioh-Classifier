import os
import requests

path = "yugioh_cards"
cards = os.listdir(path)
#print("Directory Contents: ", cards)

def getcarddata():
    url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
    id_and_frametype = []
    apiresponse = requests.get(url)

    if apiresponse.ok:
        card_data = apiresponse.json().get('data')
        for card in card_data:
            id_and_frametype.append({"id": card.get("id"), "frametype": card.get("frameType")})


        return id_and_frametype
    else:
        return None

def sortcards():

    for card in getcarddata():
        x = 0




print(getcarddata())