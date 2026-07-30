import os
import shutil
import splitfolders
import requests


def getcarddata():
    url = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
    id_and_frametype = []
    apiresponse = requests.get(url)

    card_data = apiresponse.json().get('data')
    for card in card_data:
        id_and_frametype.append({"id": card.get("id"), "frametype": card.get("frameType")})

    return id_and_frametype


def sortcards():
    path = "yugioh_cards"
    id_and_frametype = getcarddata()
    cards_with_extention = os.listdir(path)
    cards = [os.path.splitext(card)[0] for card in cards_with_extention]

    card_types = ['normal','effect','ritual','fusion','synchro','xyz','link','spell','trap']

    for card_type in card_types:
        if os.path.exists(card_type) != True:
            os.mkdir(card_type)

    if os.path.exists('misc') != True:
        os.mkdir('misc')


    counter = 0

    for dictionary in id_and_frametype:
        current_card = str(dictionary['id'])
        current_type = (dictionary['frametype'])
        print('current card: ', current_card)

        for card in cards:
            path_name = card + ".jpg"
            if card == current_card:
                print("Match!")
                counter = counter + 1
                print(counter, " Cards Matched")

                if current_type in card_types:
                    shutil.copyfile(f"{path}/{path_name}", f"{current_type}/{path_name}")

                else:
                    shutil.copyfile(f"{path}/{path_name}", f"misc/{path_name}")

    return id_and_frametype

def splitcards():
    path = "C:/Users/Robyn/Desktop/Research Project/sorted_cards"
    splitfolders.ratio(path, output="cards_split", ratio=(0.6, 0.2, 0.2))


splitcards()