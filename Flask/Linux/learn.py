from flask import Flask, jsonify, request, render_template, redirect

learnList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J', 'K', 'L', 'N', 'O', 'P', 'Q', 'S', 'T', 'U', 'X']

class LearnUI:
    def learn(self, currentSign):
        if currentSign in learnList:
            if 0 < learnList.index(currentSign) < len(learnList)-1:
                prev = learnList[learnList.index(currentSign) - 1]
                nex = learnList[learnList.index(currentSign) + 1]
            elif learnList.index(currentSign) == 0:
                prev = currentSign
                nex = learnList[learnList.index(currentSign) + 1]
            else:
                prev = learnList[learnList.index(currentSign) - 1]
                nex = currentSign
        else:
            currentSign = '1'
            prev = '1'
            nex = '2'
        learn_image = currentSign + '.png'
        return render_template("learn.html", symbol=currentSign, previous=prev, next=nex, learn_image=learn_image)