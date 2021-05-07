import math
import mediapipe as mp
import cv2
from google.protobuf.json_format import MessageToDict
import keras

# Paths where the models are saved
rightSavePath = 'static/model/alphabetaRight.h5'
leftSavePath = 'static/model/alphabetaLeft.h5'

# Loading the models
modelLeft = keras.models.load_model(leftSavePath)
modelRight = keras.models.load_model(rightSavePath)

# Model output to actual result converter dictionary
left_result_dict = {'0': 'A', '1':'B', '2': 'C', '3': 'D', '4': "F", '5': "G", '6': "J", '7': "N", '8': "Q", '9': "R", '10': 'S', '11': 'X'}

# Model output to actual result converter dictionary
right_result_dict = {'0': 'A','1': 'B','2': 'C','3': 'E','4': "F",'5': "G", '6': "I",'7': "J",'8': "K",'9': "L",'10': 'N','11': 'P','12': 'S','13': "U", '14': '0','15': '1','16': '2','17': '3','18': '4','19': '5','20': '6','21': '7', '22': '8','23': '9' }

# Initializing the MediaPipe hand tracking model
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# Defining a classifier class
class Classifier:
    # region: Member variables
    # mediaPipe configuration hands object
    __mpHands = mp.solutions.hands
    # mediaPipe detector objet
    __mpHandDetector = None

    # On initializing, this method will run
    def __init__(self):
        self.__setDefaultHandConfiguration()

    # Setting the default hand configurations for the created instance
    def __setDefaultHandConfiguration(self):
        self.__mpHandDetector = self.__mpHands.Hands(
            # default = 2
            max_num_hands=2,
            # Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the hand landmarks to be considered tracked successfully (default= 0.5)
            min_detection_confidence=0.5,
            # Minimum confidence value ([0.0, 1.0]) from the hand detection model for the detection to be considered successful. (default = 0.5)
            min_tracking_confidence=0.5
        )

    # Method that detects the signs
    def detectHands(self, capture):
        # if the MediaPipe model is not loaded, exit from the method
        if self.__mpHandDetector is None:
            return
        image = capture

        # flipping the image across the vertical axis
        image = cv2.flip(image, 1)

        # To improve performance, marking the image as not writeable to pass by reference
        image.flags.writeable = False

        # Passing the image for hand landmark tracking
        detectorResults = self.__mpHandDetector.process(image)

        # changing the writeable flag back
        image.flags.writeable = True

        # Initializing the list that will contain the prediction results
        predictions_result = []

        # If hands are detected
        if detectorResults.multi_hand_landmarks:
            # Initializing a list which contain which hand is at which position
            # i.e. if right hand is first result and left hand is second then whichhand will be ['Right', 'Left']
            whichhand = []

            # loop to go through all the hands detected to find if they are Right hand or Left hand
            for idx, hand_handedness in enumerate(detectorResults.multi_handedness):
                # converting the returned result object to a dictionary
                handedness_dict = MessageToDict(hand_handedness)

                # adding the handedness (Right or Left) to whichhand list
                whichhand.append(handedness_dict["classification"][0]["label"])

            # total number of detected hands
            number_of_hands = len(detectorResults.multi_hand_landmarks)

            # Now adding the returned hand landmark point result(s) with
            # it's respective handedness to prediction_result list
            for i in range(number_of_hands):
                # Calling the method that classifies the hand landmark points into it's respective classes
                predictions_result.append(self.simpleGesture(detectorResults.multi_hand_landmarks[i], whichhand[i]))

        # If the model predicted a sign, print it on the console
        if predictions_result:
            print(predictions_result)

        # initializing a string variable that will containt the detection result to be returned
        str_result = ""

        # printing the handedness on the console
        if predictions_result:
            print(whichhand)

        # Initializing a list that contains the signs that needs both hands
        non_one_hand = ["A", "B", "D", "E", "F", "G", "J", "K", "N", "P", 'Q', 'R', 'S', 'T', 'X']

        # if only one hand is detected
        if len(predictions_result) == 1:
            # if the hand detected is left, calling the right hand model dict because the image is flipped by webcam
            if whichhand[0] == "Left":  # HUMAN RIGHT HAND
                if str(predictions_result[0][0]) in right_result_dict.keys():
                    # adding the real meaning of the classification result to str_result
                    str_result = right_result_dict[str(predictions_result[0][0])]

                    # if the hand sign detected is from the signs that needs both hands, discard except B
                    if str_result in non_one_hand:
                        # because in the sign O, right hand is used and it same as when it is used in B
                        if str_result == "B":
                            str_result = "O"
                        else:
                            # Discard
                            str_result = ""

            # if the hand detected is right, calling the left hand model dict because the image is flipped by webcam
            elif whichhand[0] == "Right":  # HUMAN LEFT HAND
                if str(predictions_result[0][0]) in left_result_dict.keys():
                    # adding the real meaning of the classification result to str_result
                    str_result = left_result_dict[str(predictions_result[0][0])]
                    # if the hand sign detected is from the signs that needs both hands, discard it
                    if str_result in non_one_hand:
                        str_result = ""

        # if two hands are detected
        if len(predictions_result) == 2:
            # if the hand detected by mediapipe is right, it means left cause the image from webcam is flipped
            if whichhand[0] == "Right": # if first result is right, then next result will be left
                # adding the real meaning of the left and right hand result in their respective variables
                if str(predictions_result[0][0]) in left_result_dict.keys():
                    # left hand result
                    left_result = left_result_dict[str(predictions_result[0][0])]
                    if str(predictions_result[1][0]) in right_result_dict.keys():
                        # right hand result
                        right_result = right_result_dict[str(predictions_result[1][0])]

                        # comparing both hands result to construct final detection
                        if left_result == right_result:
                            if left_result in non_one_hand and right_result in non_one_hand:
                                str_result = left_result

                        # since in sign D, right hand is similar to C while left hand is unique D
                        elif left_result == 'D' and right_result == "C":
                            str_result = 'D'

                        # since in sign E, left hand is similar to C while right hand is unique E
                        elif left_result == 'C' and right_result == "E":
                            str_result = 'E'

                        # since in sign K, left hand is similar to D while right hand is unique K
                        elif left_result == "D" and right_result == "K":
                            str_result = 'K'

                        # since in sign P, left hand is similar to C while right hand is unique P
                        elif left_result == "C" and right_result == "P":
                            str_result = 'P'

                        # since in sign Q, right hand is similar to B while left hand is unique Q
                        elif right_result == 'B' and left_result == "Q":
                            str_result = "Q"

                        # since in sign R, right hand is similar to R while left hand is unique D
                        elif left_result == 'R' and right_result == 'N':
                            str_result = 'R'

                        # since in sign T, right hand is similar to E while left hand is similar to E
                        elif right_result == 'E' and left_result == 'D':
                            str_result = 'T'

                        elif left_result == 'X' and right_result == 'I':
                            str_result = 'X'

            elif whichhand[0] == "Left":
                if str(predictions_result[1][0]) in left_result_dict.keys():
                    left_result = left_result_dict[str(predictions_result[1][0])]
                    if str(predictions_result[0][0]) in right_result_dict.keys():
                        right_result = right_result_dict[str(predictions_result[0][0])]
                        # predictions_result[0] is right hand, 1 is left hand
                        # comparing both hands result to construct final detection
                        if left_result == right_result:
                            if left_result in non_one_hand and right_result in non_one_hand:
                                str_result = left_result

                        # since in sign D, right hand is similar to C while left hand is unique D
                        elif left_result == 'D' and right_result == "C":
                            str_result = 'D'

                        # since in sign E, left hand is similar to C while right hand is unique E
                        elif left_result == 'C' and right_result == "E":
                            str_result = 'E'

                        # since in sign K, left hand is similar to D while right hand is unique K
                        elif left_result == "D" and right_result == "K":
                            str_result = 'K'

                        # since in sign P, left hand is similar to C while right hand is unique P
                        elif left_result == "C" and right_result == "P":
                            str_result = 'P'

                        # since in sign Q, right hand is similar to B while left hand is unique Q
                        elif right_result == 'B' and left_result == "Q":
                            str_result = "Q"

                        # since in sign R, right hand is similar to R while left hand is unique D
                        elif left_result == 'R' and right_result == 'N':
                            str_result = 'R'

                        # since in sign T, right hand is similar to E while left hand is similar to E
                        elif right_result == 'E' and left_result == 'D':
                            str_result = 'T'

                        # since in sign X, right hand is similar to I while left hand is unique X
                        elif left_result == 'X' and right_result == 'I':
                            str_result = 'X'

        return str_result

    def simpleGesture(self, hand_landmarks, whichhand):
        # putting the different hand landmark point x coordinates into easily usable variables
        wristx = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x  # 0
        t0x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x  # 1
        t1x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x  # 2
        t2x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x  # 3
        t3x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x  # 4
        i0x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x  # 5
        i1x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x  # 6
        i2x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x  # 7
        i3x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x  # 8
        m0x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x  # 9
        m1x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x  # 10
        m2x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x  # 11
        m3x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x  # 12
        r0x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x  # 13
        r1x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x  # 14
        r2x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x  # 15
        r3x = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x  # 16
        p0x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x  # 17
        p1x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x  # 18
        p2x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x  # 19
        p3x = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x  # 20

        # putting the different hand landmark point y coordinates into easily usable variables
        wristx = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y  # 0
        t0y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y  # 1
        t1y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y  # 2
        t2y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y  # 3
        t3y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x  # 4
        i0y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y  # 5
        i1y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y  # 6
        i2y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y  # 7
        i3y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y  # 8
        m0y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y  # 9
        m1y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y  # 10
        m2y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y  # 11
        m3y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y  # 12
        r0y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y  # 13
        r1y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y  # 14
        r2y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y  # 15
        r3y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y  # 16
        p0y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y  # 17
        p1y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y  # 18
        p2y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y  # 19
        p3y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y  # 20

        # storing the hand landmark point coordinates in a list to pass it to the classification models
        detected_points = [
                              t0x, t1x, t2x, t3x,
                              i0x, i1x, i2x, i3x,
                              m0x, m1x, m2x, m3x,
                              r0x, r1x, r2x, r3x,
                              p0x, p1x, p2x, p3x,
                              t0y, t1y, t2y, t3y,
                              i0y, i1y, i2y, i3y,
                              m0y, m1y, m2y, m3y,
                              r0y, r1y, r2y, r3y,
                              p0y, p1y, p2y, p3y
        ]

        if whichhand == "Left":  # image is flipped by webcam
            # calling the right hand classification model for classification into appropriate sign
            predictions = modelRight.predict_classes([detected_points])
            return list(predictions)

        if whichhand == "Right":  # image is flipped by webcam
            # calling the left hand classification model for classification into appropriate sign
            predictions = modelLeft.predict_classes([detected_points])
            return list(predictions)

        # if no result returned, return the empty list
        return list([])
