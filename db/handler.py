import sqlite3
import os

class DBhandler:
    def __init__(self, path='./models/key_book.db'):
        self.path = path

    def connect(self):
        return sqlite3.connect(self.path)

    def get_face_tables(self):
        conn = self.connect()
        c = conn.cursor()
        listOfTables = c.execute(
            """SELECT * FROM sqlite_master WHERE type='table' 
            AND name='facials'; """).fetchall()
        conn.close()
        return listOfTables

    def create_facials_table(self):
        conn = self.connect()
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS facials (
                        facePart text
                    )""")
        conn.commit()
        conn.close()

    def insert_face(self, facePart):
        conn = self.connect()
        c = conn.cursor()
        c.execute("INSERT INTO facials VALUES (?)", (facePart,))
        conn.commit()
        conn.close()

    def create_keys_table(self):
        conn = self.connect()
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS keys (
                        descrip text,
                        hotkeys text
                    )""")
        conn.commit()
        conn.close()

    def get_all_keys(self):
        conn = self.connect()
        c = conn.cursor()
        c.execute("SELECT *, oid FROM keys")
        records = c.fetchall()
        conn.close()
        return records

    def insert_key(self, descrip, hotkeys):
        conn = self.connect()
        c = conn.cursor()
        c.execute("INSERT INTO keys VALUES (?, ?)", (descrip, hotkeys))
        conn.commit()
        conn.close()

    def update_key(self, record_id, descrip, hotkeys):
        conn = self.connect()
        c = conn.cursor()
        c.execute("""UPDATE keys SET
                        descrip = :descrip,
                        hotkeys = :hotkeys
                     WHERE oid = :oid""",
                  {
                      'descrip': descrip,
                      'hotkeys': hotkeys,
                      'oid': record_id
                  })
        conn.commit()
        conn.close()


    def delete_key(self, record_id):
        conn = self.connect()
        c = conn.cursor()
        c.execute("DELETE FROM keys WHERE oid = ?", (record_id,))
        conn.commit()
        conn.close()


    def restore_all(self):
        conn = self.connect()
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS facials;")
        conn.commit()
        c.execute("DROP TABLE IF EXISTS domains;")
        conn.commit()
        c.execute("DROP TABLE IF EXISTS positions;")
        conn.commit()
        conn.close()

