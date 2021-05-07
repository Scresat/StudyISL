from flask import Flask, jsonify, request, render_template, redirect
from imageio import imread
import math
import mediapipe as mp
import cv2
from google.protobuf.json_format import MessageToDict
from classifier import Classifier


class classifierCommunicator:
    cls = Classifier()

    def detection(self):
        file = request.files['webcam']
        image_np = imread(file)

        disp_str = self.cls.detectHands(image_np)
        if disp_str == "":
            disp_str = 'No sign detected'

        display_str = {'name': disp_str}

        return jsonify(display_str)