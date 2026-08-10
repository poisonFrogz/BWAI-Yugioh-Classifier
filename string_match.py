from fuzzywuzzy import fuzz
from fuzzywuzzy import process


def find_string(path,prediction,card_type):

    cards_to_search = []

    if prediction == 'None':
        best_match = 'OCR Error'
        print(best_match)
    else:
        with open(f'{path}{card_type}.txt','r') as f:
            cards_to_search = f.read().splitlines()
            f.close()

        best_match, score = process.extractOne(prediction, cards_to_search, scorer = fuzz.token_sort_ratio)
        print(best_match,score)
        if score < 90:
            with open(f'{path}all_cards.txt','r') as f:
                cards_to_search = f.read().splitlines()
                f.close()

            best_match, score = process.extractOne(prediction, cards_to_search, scorer = fuzz.token_sort_ratio)
            print(best_match,score)

            if score < 90:
                best_match = "Can't Predict"


    return best_match

path = 'C:/Users/Robyn/Desktop/Research Project/card_types/'
prediction = "WINGED DRAGON OF RA"
card_type = 'xyz'
find_string(path,prediction,card_type)