import sqlite3


class Communicator:
    # connect to the database
    conn = sqlite3.connect('learn.db')

    # getting a cursor that will be used to execute the sql commands
    c = conn.cursor()

    # function that returns a list of sign that can be learnt from the platform
    def get_learn_list(self):
        self.c.execute('SELECT * FROM learn_table')
        return self.c.fetchall()


dbconnect = Communicator()