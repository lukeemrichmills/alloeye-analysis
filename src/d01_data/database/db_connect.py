import psycopg2
from src.d01_data.database.config_psql import config


def db_connect(section='local', suppress_print=False):
    """ Connect to the PostgreSQL database server (does not disconnect)"""
    conn = None
    try:
        # read connection parameters
        params = config(section)

        # connect to the PostgreSQL server
        if not suppress_print:
            print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        if not suppress_print:
            print('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        if not suppress_print:
            print(db_version)

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    return conn


if __name__ == '__main__':
    db_connect()
