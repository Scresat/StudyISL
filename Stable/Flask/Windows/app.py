from flask import Flask
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


if __name__ == '__main__':
    app.run()




