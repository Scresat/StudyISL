from flask import render_template


class quiz:
    quizlist = [
        {"value": ["0", "1", "2", "3", "4"], "title": "Numbers quiz 1", "id": 1},
        {"value": ["5", "6", "7", "8", "9"], "title": "Numbers quiz 2", "id": 2},
        {"value": ["A", "B", "C", "D"], "title": "Alphabets quiz 1", "id": 3},
        {"value": ["E", "F", "G", "I"], "title": "Alphabets quiz 2", "id": 4},
        {"value": ["J", "K", "L", "N"], "title": "Alphabets quiz 3", "id": 5},
        {"value": ["O", "P", "Q", "S"], "title": "Alphabets quiz 4", "id": 6},
        {"value": ["T", "U", "V", "W"], "title": "Alphabets quiz 5", "id": 7},
        {"value": ["W", "U", "X", "Z"], "title": "Alphabets quiz 6", "id": 8}
    ]

    def return_quizpage(self):
        return render_template("quizpage.html", quizlist=self.quizlist)

    def return_quiz(self, id):
        current_quizlist = self.quizlist[0]["value"]
        for i in self.quizlist:
            if id == i["id"]:
                current_quizlist = i["value"]
        print(current_quizlist)
        return render_template("quiz.html", quiz_list=current_quizlist)
