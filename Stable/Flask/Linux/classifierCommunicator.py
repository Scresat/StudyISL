from flask import jsonify, request
from imageio import imread
from classifier import Classifier


class classifierCommunicator:
    # creating a classifier object used for classification
    cls = Classifier()

    def detection(self):
        # access the image file sent from client browser
        file = request.files['webcam']
        image_np = imread(file)

        # call the detect method that return the result of classification
        disp_str = self.cls.detectHands(image_np)

        # if no sign found then return "No sign detected"
        if disp_str == "":
            disp_str = 'No sign detected'

        # else if sign is detected return that sign
        display_str = {'name': disp_str}

        return jsonify(display_str)