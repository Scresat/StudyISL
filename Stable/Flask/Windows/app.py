from flask import Flask, render_template
from classifierCommunicator import classifierCommunicator
from learn import LearnUI
from home import Home
from quiz import quiz


app = Flask(__name__)


lComm = classifierCommunicator()
lUI = LearnUI()
homePage = Home()
quizObj = quiz()


@app.route('/')  # route for homepage
def homepage():
    return homePage.homepage()


@app.route('/learn/<symbol>')  # route for learn page
def learn(symbol):
    return lUI.learn(symbol)


@app.route('/detect', methods=["POST"])  # route for sign detection API that return the sign detected
def detection():
    return lComm.detection()


@app.route('/quizpage')
def quizpage():
    return quizObj.return_quizpage()

@app.route('/quiz/<id>')
def quiz(id):
    return quizObj.return_quiz(int(id))

if __name__ == '__main__':
    app.run()




