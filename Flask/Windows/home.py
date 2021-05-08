from flask import render_template
from communication import dbconnect

# Return a homepage
class Home:
    # get the list of signs from the database that are available to learn on the platform
    learnsymbols = dbconnect.get_learn_list()
    learnList = []

    for i in range(len(learnsymbols)):
        learnList.append(learnsymbols[i][0])

    # pass the list of signs to the html template for further processing
    def homepage(self):
        return render_template("home.html", learnList=self.learnList)