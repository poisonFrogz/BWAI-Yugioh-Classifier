import cv2

def extract_text(card, reader):
    card_path = card
    card_to_extract = cv2.imread(card_path)
    card_to_gray = cv2.cvtColor(card_to_extract, cv2.COLOR_BGR2GRAY)
    card_resized = cv2.resize(card_to_gray, (1000, 1000), interpolation = cv2.INTER_CUBIC)
    card_cropped = card_resized[78:300,:887]

    extract_info = reader.readtext(card_cropped)
    ocr_output = []

    for bbox, text, prob in extract_info:
        if prob > 0.5:
            ocr_output.append(text)

    if ocr_output:
        prediction = " ".join(ocr_output)
        print(prediction)
    else:
        prediction = "None"
        print(prediction)

    return prediction
