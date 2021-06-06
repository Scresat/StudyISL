from flask import Flask, render_template
from classifierCommunicator import classifierCommunicator
from learn import LearnUI
from home import Home


app = Flask(__name__)


lComm = classifierCommunicator()
lUI = LearnUI()
homePage = Home()


@app.route('/')  # route for homepage
def homepage():
    return homePage.homepage()


@app.route('/learn/<symbol>')  # route for learn page
def learn(symbol):
    return lUI.learn(symbol)


@app.route('/detect', methods=["POST"])  # route for sign detection API that return the sign detected
def detection():
    return lComm.detection()

@app.route('/quiz')
def quiz():
    quiz_list = ["A", "B", "C", "D"]
    return render_template("quiz.html", quiz_list=quiz_list)

if __name__ == '__main__':
    app.run()




