import sqlite3

conn = sqlite3.connect('learn.db')

c = conn.cursor()

c.execute('DROP TABLE IF EXISTS learn_table')
conn.commit()

c.execute('CREATE TABLE learn_table (learn text, learn_image text)')
conn.commit()

learnList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J', 'K',
                 'L', 'N', 'O', 'P', 'Q', 'S', 'T', 'U', 'V', 'X']

for i in learnList:
    c.execute('''INSERT INTO learn_table values (:learn, :learn_image)''', {'learn': i, 'learn_image': i+'.png'})
    conn.commit()

c.execute('select * from learn_table')
print(c.fetchall())

conn.close()