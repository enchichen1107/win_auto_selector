import sqlite3

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
        c.execute("""
            UPDATE keys
            SET descrip = ?, hotkeys = ?
            WHERE oid = ?
        """, (descrip, hotkeys, record_id))
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

    def get_face_part(self):
        conn = self.connect()
        c = conn.cursor()
        c.execute("SELECT facePart FROM facials")
        record = c.fetchall()
        conn.close()
        return record
    
    def get_positions_tables(self):
        conn = self.connect()
        c = conn.cursor()
        listOfTables = c.execute(
            """SELECT * FROM sqlite_master WHERE type='table' 
            AND name='positions'; """).fetchall()
        conn.close()
        return listOfTables

    def create_positions_table(self):
        conn = self.connect()
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS positions (
            pos INT
            )""")
        conn.commit()
        conn.close()

    def insert_face_pos(self, pos):
        conn = self.connect()
        c = conn.cursor()
        c.execute("INSERT INTO positions VALUES (?)", (pos,))
        conn.commit()
        conn.close()

    def update_face_pos(self, pos):
        conn = self.connect()
        c = conn.cursor()
        c.execute("UPDATE positions SET pos = ?", (pos,))
        conn.commit()
        conn.close()
    
    def get_face_pos(self):
        conn = self.connect()
        c = conn.cursor()
        c.execute("SELECT pos FROM positions")
        record = c.fetchall()
        conn.close()
        return record
    
    def domain_increment(self):
        """Get current domain count and update it."""
        conn = self.connect()
        c = conn.cursor()

        # Check if 'domains' table exists
        listOfTables = c.execute(
            """SELECT * FROM sqlite_master WHERE type='table' 
            AND name='domains';"""
        ).fetchall()

        # If not exists → create and initialize
        if not listOfTables:
            c.execute("""CREATE TABLE IF NOT EXISTS domains (
                            id INTEGER,
                            counts INTEGER
                        )""")
            conn.commit()

            # Initialize domain count
            c.execute("INSERT INTO domains VALUES (?, ?)", (1, 1))
            conn.commit()

        else:
            # Get current count
            c.execute("SELECT counts FROM domains WHERE id = 1")
            record = c.fetchone()
            if record:
                new_counts = record[0] + 1
                # Update count
                c.execute("UPDATE domains SET counts = ? WHERE id = ?", (new_counts, 1))
                conn.commit()

        conn.close()

    def get_domain_count(self):
        """Return current domain count (from domains table)."""
        conn = self.connect()
        c = conn.cursor()
        c.execute("SELECT counts FROM domains WHERE id = 1")
        record = c.fetchone()
        conn.close()
        return record[0] if record else 0


