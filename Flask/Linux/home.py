from flask import Flask, jsonify, request, render_template, redirect, views
from communication import dbconnect

class Home:
    learnsymbols = dbconnect.get_learn_list()
    learnList = []

    for i in range(len(learnsymbols)):
        learnList.append(learnsymbols[i][0])

    def homepage(self):
        return render_template("home.html", learnList=self.learnList)