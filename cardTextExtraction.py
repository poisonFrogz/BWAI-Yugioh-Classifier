import cv2

def extract_text(card, reader):
    card_path = card
    card_to_extract = cv2.imread(card_path)
    card_to_rgb = cv2.cvtColor(card_to_extract, cv2.COLOR_BGR2RGB)
    card_to_gray = cv2.cvtColor(card_to_extract, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    card_contrast = clahe.apply(card_to_gray)
    card_resized = cv2.resize(card_contrast, (2000, 2000), interpolation = cv2.INTER_CUBIC)
    card_cropped = card_resized[:300,:]
    card_gaussian = cv2.GaussianBlur(card_cropped, (5,5), 0)


    cv2.imshow("Card", card_cropped)
    cv2.waitKey(0)


    extract_info = reader.readtext(card_cropped)
    ocr_output = []

    for bbox, text, prob in extract_info:
        if prob > 0.4:
            ocr_output.append(text)

    if ocr_output:
        prediction = " ".join(ocr_output)
        #print(prediction)
    else:
        prediction = "None"
        #print(prediction)

    return prediction
