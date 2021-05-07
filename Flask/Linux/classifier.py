import math
import mediapipe as mp
import cv2
import tflite_runtime.interpreter as tflite
from google.protobuf.json_format import MessageToDict
import numpy as np

i = 0
interpreterLeft = tflite.Interpreter(model_path="static/model/modelLeft.tflite")
interpreterLeft.allocate_tensors()

interpreterRight = tflite.Interpreter(model_path="static/model/modelRight.tflite")
interpreterRight.allocate_tensors()

input_detailsLeft = interpreterLeft.get_input_details()
output_detailsLeft = interpreterLeft.get_output_details()

input_detailsRight = interpreterRight.get_input_details()
output_detailsRight = interpreterRight.get_output_details()

left_result_dict = {'0': 'A', '1':'B', '2': 'C', '3': 'D', '4': "F", '5': "G", '6': "J", '7': "N", '8': "Q", '9': "R", '10': 'S', '11': 'X'}

right_result_dict = {'0': 'A', 
                     '1': 'B', 
                     '2': 'C', 
                     '3': 'E', 
                     '4': "F", 
                     '5': "G", 
                     '6': "I", 
                     '7': "J", 
                     '8': "K", 
                     '9': "L", 
                     '10': 'N', 
                     '11': 'P', 
                     '12': 'S', 
                     '13': "U", 
                     '14': '0', 
                     '15': '1', 
                     '16': '2',
                     '17': '3',
                     '18': '4',
                     '19': '5',
                     '20': '6',
                     '21': '7',
                     '22': '8',
                     '23': '9'
                    }
#right c, b, z

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


class Classifier:
    # region: Member variables
    # mediaPipe configuration hands object
    __mpHands = mp.solutions.hands
    # mediaPipe detector objet
    __mpHandDetector = None

    def __init__(self):
        self.__setDefaultHandConfiguration()

    def __setDefaultHandConfiguration(self):
        self.__mpHandDetector = self.__mpHands.Hands(
            # default = 2
            max_num_hands=2,
            # Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the hand landmarks to be considered tracked successfully (default= 0.5)
            min_detection_confidence=0.5,
            # Minimum confidence value ([0.0, 1.0]) from the hand detection model for the detection to be considered successful. (default = 0.5)
            min_tracking_confidence=0.5
        )

    def detectHands(self, capture):
        if self.__mpHandDetector is None:
            return
        image = capture
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        detectorResults = self.__mpHandDetector.process(image)
        image.flags.writeable = True
        predictions_result = []

        if detectorResults.multi_hand_landmarks:
            whichhand = []
            for idx, hand_handedness in enumerate(detectorResults.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                # print(hand_handedness)
                whichhand.append(handedness_dict["classification"][0]["label"])

            number_of_hands = len(detectorResults.multi_hand_landmarks)

            # for hand_landmarks in detectorResults.multi_hand_landmarks:
            for i in range(number_of_hands):
                predictions_result.append(self.simpleGesture(detectorResults.multi_hand_landmarks[i], whichhand[i]))
                mp_drawing.draw_landmarks(image, detectorResults.multi_hand_landmarks[i], mp_hands.HAND_CONNECTIONS)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        font = cv2.FONT_HERSHEY_SIMPLEX

        if predictions_result != []:
            print(predictions_result)

        

        str_result = ""
        if predictions_result != []:
            print(whichhand)

        non_one_hand = ["A", "B", "D", "E", "F", "G", "J", "K", "N", "P", 'Q', 'R', 'S', 'T', 'X']
        if len(predictions_result) == 1:
            if whichhand[0] == "Left":  # HUMAN RIGHT HAND
                if str(predictions_result[0][0]) in right_result_dict.keys():
                    str_result = right_result_dict[str(predictions_result[0][0])]
                    if str_result in non_one_hand:
                        if str_result == "B":
                            str_result = "O"
                        else:
                            str_result = ""

            elif whichhand[0] == "Right":  # HUMAN LEFT HAND
                if str(predictions_result[0][0]) in left_result_dict.keys():
                    str_result = left_result_dict[str(predictions_result[0][0])]
                    if str_result in non_one_hand:
                        str_result = ""

        # print(whichhand)
        if len(predictions_result) == 2:
            if str(predictions_result[0][0]) == "20" and str(predictions_result[1][0]) == "20":
                str_result == "7"
            if whichhand[0] == "Right":
                if str(predictions_result[0][0]) in left_result_dict.keys():
                    left_result = left_result_dict[str(predictions_result[0][0])]
                    if str(predictions_result[1][0]) in right_result_dict.keys():
                        right_result = right_result_dict[str(predictions_result[1][0])]
                        # predictions_result[0] is right hand, 1 is left hand
                        if left_result == right_result:
                            if left_result in non_one_hand and right_result in non_one_hand:
                                str_result = left_result
                        elif left_result == 'D' and right_result == "C":
                            str_result = 'D'
                        elif left_result == 'C' and right_result == "E":
                            str_result = 'E'
                        elif left_result == "D" and right_result == "K":
                            str_result = 'K'
                        elif left_result == "C" and right_result == "P":
                            str_result = 'P'
                        elif right_result == 'B' and left_result == "Q":
                            str_result = "Q"
                        elif left_result == 'R' and right_result == 'N':
                            str_result = 'R'
                        elif left_result == 'Q' and right_result == 'N':
                            str_result = 'R'
                        elif left_result == 'N' and right_result == 'P':
                            str_result = 'R'
                        elif left_result == 'P' and right_result == 'N':
                            str_result = 'R'
                        elif right_result == 'K' and left_result == 'N':
                            str_result = 'K'
                        elif right_result == 'E' and left_result == 'D':
                            str_result = 'T'
                        elif right_result == 'G' and left_result == 'C':
                            str_result = 'Y'
                        elif right_result == 'G' and left_result == 'J':
                            str_result = 'J'
                        elif right_result == 'I' and left_result == 'G':
                            str_result = 'J'
                        elif right_result == 'E' and left_result == 'X':
                            str_result = 'X'
                        elif left_result == 'X' and right_result == 'I':
                            str_result = 'X'

            elif whichhand[0] == "Left":
                if str(predictions_result[1][0]) in left_result_dict.keys():
                    left_result = left_result_dict[str(predictions_result[1][0])]
                    if str(predictions_result[0][0]) in right_result_dict.keys():
                        right_result = right_result_dict[str(predictions_result[0][0])]
                        # predictions_result[0] is right hand, 1 is left hand
                        if left_result == right_result:
                            str_result = left_result
                        elif left_result == 'D' and right_result == "C":
                            str_result = 'D'
                        elif left_result == 'C' and right_result == "E":
                            str_result = 'E'
                        elif left_result == "D" and right_result == "K":
                            str_result = 'K'
                        elif left_result == "C" and right_result == "P":
                            str_result = 'P'
                        elif right_result == 'B' and left_result == "Q":
                            str_result = "Q"
                        elif left_result == 'R' and right_result == 'N':
                            str_result = 'R'
                        elif left_result == 'Q' and right_result == 'N':
                            str_result = 'R'
                        elif left_result == 'N' and right_result == 'P':
                            str_result = 'R'
                        elif left_result == 'P' and right_result == 'N':
                            str_result = 'R'
                        elif right_result == 'K' and left_result == 'N':
                            str_result = 'K'
                        elif right_result == 'E' and left_result == 'D':
                            str_result = 'T'
                        elif right_result == 'G' and left_result == 'C':
                            str_result = 'Y'
                        elif right_result == 'G' and left_result == 'J':
                            str_result = 'J'
                        elif right_result == 'I' and left_result == 'G':
                            str_result = 'J'
                        elif right_result == 'E' and left_result == 'X':
                            str_result = 'X'
                        elif left_result == 'X' and right_result == 'I':
                            str_result = 'X'
        
        print(str_result)
        return str_result

    def simpleGesture(self, hand_landmarks, whichhand):
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

        input_data = np.array([[t0x, t1x, t2x, t3x, i0x, i1x, i2x, i3x, m0x, m1x, m2x, m3x, r0x, r1x, r2x, r3x, p0x, p1x, p2x, p3x, t0y, t1y, t2y, t3y, i0y, i1y, i2y, i3y, m0y, m1y, m2y, m3y, r0y, r1y, r2y, r3y, p0y, p1y, p2y, p3y]], dtype=np.float32)

        if whichhand == "Left":
            #predictions = modelRight.predict_classes([[t0x, t1x, t2x, t3x, i0x, i1x, i2x, i3x, m0x, m1x, m2x, m3x, p0x, p1x, p2x, p3x, t0y, t1y, t2y, t3y, i0y, i1y, i2y, i3y, m0y, m1y, m2y, m3y, p0y, p1y, p2y, p3y]])
#             predictions = list(predictions)
    
#             #print(predictionsdict[predictions[0]], "Right")
#             str = predictionsdict[predictions[0]] + " Right"
            
            interpreterRight.set_tensor(input_detailsRight[0]['index'], input_data)
            interpreterRight.invoke()
            output_data = interpreterRight.get_tensor(output_detailsRight[0]['index'])
            output_data = list(output_data[0])
            predictions = output_data.index(max(output_data))
            result = str(predictions)
            result = [result]
            print(predictions)
            return result
        if whichhand == "Right":
            #predictions = modelLeft.predict_classes([[t0x, t1x, t2x, t3x, i0x, i1x, i2x, i3x, m0x, m1x, m2x, m3x, p0x, p1x, p2x, p3x, t0y, t1y, t2y, t3y, i0y, i1y, i2y, i3y, m0y, m1y, m2y, m3y, p0y, p1y, p2y, p3y]])
#             predictions = list(predictions)
            
#             #print(predictionsdict[predictions[0]], "Left")
#             str = predictionsdict[predictions[0]] + " Left"
            interpreterLeft.set_tensor(input_detailsLeft[0]['index'], input_data)
            interpreterLeft.invoke()
            output_data = interpreterLeft.get_tensor(output_detailsLeft[0]['index'])
            output_data = list(output_data[0])
            predictions = output_data.index(max(output_data))
            result = str(predictions)
            result = [result]
            print(predictions)
            return result
        return list(str)
