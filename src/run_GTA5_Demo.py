import numpy as np
import os, cv2, time, mss
from NvidiaNet import *
import pygetwindow
import pyautogui
from directkeys import PressKey, W, A, S, D


def resizeImage(img): #make sure that each input image to the model is of the same dimensions
    '''
    :param img:
    :return: resizedImg : Size = [1,1,160,120] # Do Not Change
    '''
    greyscaleImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resizedImg = cv2.resize(greyscaleImg, (160, 120))
    resizedImg = resizedImg.reshape(1,1,WIDTH,HEIGHT)

    return resizedImg

def test_ToTensor(img): #convert image into a tensor
    '''
    :param img:
    :return img Tensor
    '''
    return torch.from_numpy(img)

def moveCar(prediction):
    new_prediction = prediction.detach().numpy()[0]
    if (-0.3 < new_prediction[0] < 0.3  ):
        PressKey(W)
    elif (-1.0 < new_prediction[0] < -0.3  ):
        PressKey(W)
        PressKey(A)
    elif (0.3 < new_prediction[0] < 1.0 ):
        PressKey(W)
        PressKey(D)


# -------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    '''
    Initiate COUNTDOWN.
    Switch to GTA 5 window when countdown begins.
    '''
    print("COUNTDOWN begins...")
    print("Please switch to GTA 5 window.")
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)
    ################################################################################################################
    '''                                               Import Trained Model                                       '''
    ################################################################################################################

    net_nvidia = Net().to(device)
    model_name = 'kaggle_jigsaw_vatsal_iter_1.pth'
    path = f"../models/{model_name}"
    net_nvidia.load_state_dict(torch.load(path))


    ################################################################################################################
    '''                                              Set Project Variables                                       '''
    ################################################################################################################

    # Image variables
    NEW_SIZE = (400, 220)
    WIDTH = 160
    HEIGHT = 120

    previousTime = time.time()

    ################################################################################################################
    '''                                                   GTA 5 Demo                                             '''
    ################################################################################################################
    ENABLE_AUTONOMOUS_MODE = True
    print("Running...")
    with mss.mss() as sct:
        while(True):

            #Check for keyboard interrupt
            try:
                start = time.time()

                #Get GTA 5 Screen dimensions using pygetwindow.
                GTA_Screen = pygetwindow.getWindowsWithTitle("Grand Theft Auto V")[0]
                monitor = {"top": GTA_Screen.top, "left": GTA_Screen.left,
                           "width": GTA_Screen.width, "height": GTA_Screen.height}

                # Capture the GTA5 window.
                screen = np.array(sct.grab(monitor))
                screen = cv2.resize(screen, NEW_SIZE)
                # Display Captured window.
                cv2.imshow("win", screen)
                # Convert image color format from BGR to RGB.
                screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
                cv2.waitKey(10)

                '''
                If ENABLE_AUTONOMOUS_MODE flag is True:
                    Generate Steering Angle and Throttle prediction from screen capture using Trained model.
                '''
                if ENABLE_AUTONOMOUS_MODE :
                    resizedImage = resizeImage(screen)
                    test_tensor2 = test_ToTensor(resizedImage)
                    input = test_tensor2.to(device, dtype=torch.float)
                    pred_output = net_nvidia(input)
                    print("Prediction: ",pred_output)
                    moveCar(pred_output)

            except KeyboardInterrupt:
                print("Closing")
                break


    cv2.destroyAllWindows()







