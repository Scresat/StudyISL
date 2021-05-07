from flask import Flask
from classifierCommunicator import classifierCommunicator
from learn import LearnUI
from home import Home


app = Flask(__name__)


lComm = classifierCommunicator()
lUI = LearnUI()
homePage = Home()


@app.route('/')
def homepage():
    return homePage.homepage()


@app.route('/learn/<symbol>')
def learn(symbol):
    return lUI.learn(symbol)


@app.route('/detect', methods=["POST"])
def detection():
    return lComm.detection()


if __name__ == '__main__':
    app.run()




