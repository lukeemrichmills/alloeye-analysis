import psycopg2

from src.d01_data.database.Errors import InvalidSQLOutput, InvalidValue
from src.d01_data.database.Tables import Tables


class PsqlCommander:

    def __init__(self, commands=""):
        self.commands = commands

    def run_commands(self, connection, skip_bools=None, commands=""):
        """
        :param connection: psycopg2 connector
        :param skip_bools: list of bool
            list of booleans for which commands to skip
        :return:
        """

        if commands == "":
            commands = self.commands

        err_command = None
        if skip_bools is None:
            skip_bools = [False for i in range(len(commands))]
        try:
            cursor = connection.cursor()
            # create enums one by one
            for index, command in enumerate(commands):
                err_command = command
                if skip_bools[index] is False:
                    self.execute_with_catch(cursor, command)
                else:
                    print("skipped the following command:" + command[:100])
            # close communication with the PostgreSQL database server
            cursor.close()
            # commit the changes
            connection.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            print(err_command)
        # finally:
            # if connection is not None:
                   # connection.close()

    def check_tables_exist(self, connection, check_list, skip_created=True):
        commands = self.commands
        skip_bools = [False for i in range(len(check_list))]

        exist_check_query = """
            SELECT EXISTS(
            SELECT *
            FROM information_schema.tables
            WHERE table_name=%s)
            """
        cursor = connection.cursor()
        for index, name in enumerate(check_list):
            cursor.execute(exist_check_query, (name,))
            if cursor.fetchone()[0] is True:
                print(name + " already exists")
                if skip_created:
                    skip_bools[index] = True
            else:
                print(name + " does not exist")

        return skip_bools

    def get_table_metadata(self, connection, table):
        commands = self.commands
        # CHECK IF table IS WITHIN LIST OF VALID TABLES
        table_rows_query_root = """
        SELECT COUNT(*)
        FROM """
        table_rows_query = table_rows_query_root + table
        unique_pIDs_query = "SELECT pid FROM participants"

        table_rows_output = self.fetch_query(connection, table_rows_query)
        unique_pIDs_output = self.fetch_query(connection, unique_pIDs_query)

        return [table_rows_output, unique_pIDs_output]

    def execute_query(self, connection, query="", values=None):
        if query == "":
            query = self.commands
        cursor = connection.cursor()

        self.execute_with_catch(cursor, query, values)
        # cursor.close()
        connection.commit()

    def execute_with_catch(self, cursor, command, values=None):
        if command == "":
            pass
        elif values is None:
            try:
                cursor.execute(command)
            except (Exception, psycopg2.DatabaseError) as error:
                print(error)
                print(command)
                raise error
            except (Exception, psycopg2.OperationalError) as error:
                print(error)
                print(command)
        else:
            try:
                cursor.execute(command, (values,))
            except (Exception, psycopg2.DatabaseError) as error:
                print(error)
                print(command)
                raise error
        # finally:
        #     continue

    def fetch_query(self, connection, query="", values=None):
        if query == "":
            query = self.commands
        cursor = connection.cursor()
        # UNIT TEST COMMAND INPUT - CAN'T BE MORE THAN ONE cursor.fetchall()
        if values is None:
            cursor.execute(query)
            output = cursor.fetchall()
        else:
            cursor.execute(query, (values,))
            output = cursor.fetchall()
        output_list = [row[0] for row in output]
        return output_list

    def fetch_bool_query(self, connection, query="", values=None):
        if query == "":
            query = self.commands

        result = self.fetch_query(connection, query, values)
        if len(result) > 1:
            raise InvalidSQLOutput

        return self.convertSQLbool(result[0])

    def convertSQLbool(self, sql_bool_string):
        if isinstance(sql_bool_string, bool):
            return sql_bool_string
        elif sql_bool_string != 'true' and sql_bool_string != 'false':
            raise InvalidValue
        return sql_bool_string == 'true'

    def fetch_all_table_columns(self, connection):
        fetch_table_name_query = """
                                SELECT table_name
                                FROM information_schema.tables
                                WHERE TABLE_TYPE = 'BASE TABLE'
                                AND table_schema = 'public'
                                """
        table_names = self.fetch_query(connection, fetch_table_name_query, None)
        table_column_names_dict = {}
        for table in table_names:
            table_column_names_dict[table] = PsqlCommander.fetch_table_columns(connection, table)

        return table_column_names_dict

    @staticmethod
    def fetch_table_columns(connection, table):
        """returns list of table column names for specified table,
        assumes table exists"""

        query = """
                SELECT column_name
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE table_name=%s
                ORDER BY ordinal_position
                """
        commander = PsqlCommander(query)
        column_names = commander.fetch_query(connection, "", table)

        return column_names     # without primary key row_id


