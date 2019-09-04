import os
import sys
import psycopg2

class PSQLOpration:
    DB_HOST = ""
    DB_DATABASE = ""
    DB_PORT = -1
    DB_USER = ""
    DB_PASS = ""
    def __init__(self):
        self.DB_HOST = os.environ["DB_HOST"]
        self.DB_DATABASE = os.environ["DB_DATABASE"]
        self.DB_PORT = os.environ["DB_PORT"]
        self.DB_USER = os.environ["DB_USER"]
        self.DB_PASS = os.environ["DB_PASSWORD"]
        
        if self.DB_PORT == -1:
            print("Error: There's no envirionmental variable of 'DB_PORT'.")
            sys.exit()

    def connect(self):
        pass
    def create_table(self):
        pass
    def insert(self, tbname, record):
        pass
    def select(self):
        pass
    def delete(self):
        pass
    def truncate(self):
        pass

# 継承
class RamenDB(PSQLOpration):
    def __init__(self):
        pass
    def insert(self, record):
        # アトリビュートチェックしたりしてから共通処理(ここではinsert)を親から呼び出し
        # super().insert(tbname, record)
        pass