import os
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

    monsters = ['normal','effect','ritual','fusion','synchro','xyz','link']
    if os.path.exists('monsters') != True:
        os.mkdir('monsters')

    pendulum = ['normal_pendulum','effect_pendulum','ritual_pendulum','fusion_pendulum','synchro_pendulum','xyz_pendulum']
    if os.path.exists('pendulum') != True:
        os.mkdir('pendulum')

    spell = ['spell']
    if os.path.exists('spell') != True:
        os.mkdir('spell')

    trap = ['trap']
    if os.path.exists('trap') != True:
        os.mkdir('trap')

    counter = 0

    for dictionary in id_and_frametype:
        current_card = str(dictionary['id'])
        current_type = (dictionary['frametype'])
        print('current card: ', current_card)

        for card in cards:
            path_name = card + '.jpg'
            if card == current_card:
                print("Match!")
                counter = counter + 1
                print(counter, " Cards Matched")

                for card_type in monsters:
                    if card_type == current_type:
                        os.replace("%path/%path_name", "monsters/%pathname")

                for card_type in pendulum:
                    if card_type == current_type:
                        os.replace("%path/%path_name", "pendulum/%pathname")

                for card_type in spell:
                    if card_type == current_type:
                        os.replace("%path/%path_name", "spell/%pathname")

                for card_type in trap:
                    if card_type == current_type:
                        os.replace("%path/%path_name", "trap/%pathname")














    return id_and_frametype



sortcards()
#print(sortcards())