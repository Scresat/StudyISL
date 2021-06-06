from flask import Flask, jsonify, request, render_template, redirect
from communication import dbconnect

# get the list of signs from the database that are available to learn on the platform
learnsymbols = dbconnect.get_learn_list()
learnList = []

for i in range(len(learnsymbols)):
    learnList.append(learnsymbols[i][0])

class LearnUI:
    # function that returns a unique webpage for the sign
    def learn(self, currentSign):
        # for the values of the previous and next button
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
        else:  # if sign is not in list, redirect to the first sign
            currentSign = '1'
            prev = '1'
            nex = '2'
        learn_image = currentSign + '.png'
        return render_template("learn.html", symbol=currentSign, previous=prev, next=nex, learn_image=learn_image)