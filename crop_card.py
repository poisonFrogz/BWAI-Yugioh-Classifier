import cv2

def crop_board(board_path,count,output_path):
    #load board state from path
    board_state = cv2.imread(board_path)

    #transform image to grayscale
    board_state_gray = cv2.cvtColor(board_state, cv2.COLOR_BGR2GRAY)

    #binarize image where background black and foreground white
    _, thresh = cv2.threshold(board_state_gray, 110, 225, cv2.THRESH_BINARY_INV)

    #find contours in image
    edge = cv2.Canny(thresh, 30,200)

    contours, hierarchy = cv2.findContours(edge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)




        #draw bounding box around original board image
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 100 or rect[3] < 100: continue
        x,y,w,h = rect
        #cv2.rectangle(board_state,(x,y),(x+w,y+h),(0,255,0),2)
        # save card image to output path
        roi = board_state[y:y+h,x:x+w]
        cv2.imwrite(f'{output_path}cropped_card_{count}.jpg', roi)#save image
        count = count + 1

    return count

def draw_boxes(board_path, text, output_path=0):
    # load board state from path
    board_state = cv2.imread(board_path)

    # transform image to grayscale
    board_state_gray = cv2.cvtColor(board_state, cv2.COLOR_BGR2GRAY)

    # binarize image where background black and foreground white
    _, thresh = cv2.threshold(board_state_gray, 110, 225, cv2.THRESH_BINARY_INV)

    # find contours in image
    edge = cv2.Canny(thresh, 30, 200)

    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count = 0

    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 100 or rect[3] < 100: continue
        x,y,w,h = rect
        card = cv2.rectangle(board_state,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(card, text[count],(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        count = count + 1
    c = cv2.resize(board_state, (800, 600))
    cv2.imshow("Annotated", c)
    cv2.waitKey(0)




#card_text = ['Zefra War','Token Collector','Pharaohs Treasure','Red Sprinter','Armoured Zombie',
#             'Barricadeborg Blocker','Silvers Cry','F.A. Shining Star GT',
#             'Draconic Tactics','Summoned Skull']

#path = 'C:/Users/Robyn/Desktop/Research Project/test/test_board/board_state_1.jpg'

#draw_boxes(path,card_text)