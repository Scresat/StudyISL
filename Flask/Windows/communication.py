import sqlite3

class Communicator:
    conn = sqlite3.connect('learn.db')

    c = conn.cursor()

    def get_learn_list(self):
        self.c.execute('SELECT * FROM learn_table')
        return self.c.fetchall()

dbconnect = Communicator()