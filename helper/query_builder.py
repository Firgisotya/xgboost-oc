class QueryBuilder:
    def __init__(self, model):
        self.model = model
        self.table = model.table
        self.columns = ['*']
        self.query = ''
        self.where_conditions = []
        self.order_by_column = None
        self.order_by_direction = None
        self.group_by_columns = []
        self.join_tables = []
        self.insert_data = {}
        self.update_data = {}
        self.delete_data = {}

    def select(self, columns):
        self.columns = columns
        return self

    def where(self, column, operator, value):
        condition = f"{column} {operator} {value}"
        self.where_conditions.append(condition)
        return self

    def and_where(self, column, operator, value):
        condition = f"AND {column} {operator} '{value}'"
        self.where_conditions.append(condition)
        return self
    
    def or_where(self, column, operator, value):
        condition = f"OR {column} {operator} '{value}'"
        self.where_conditions.append(condition)
        return self

    def order_by(self, column, direction='asc'):
        self.order_by_column = column
        self.order_by_direction = direction
        return self

    def group_by(self, columns):
        if isinstance(columns, str):
            self.group_by_columns.append(columns)
        elif isinstance(columns, list):
            self.group_by_columns.extend(columns)
        return self

    def join(self, table, on_condition):
        self.join_tables.append((table, on_condition))
        return self

    def get(self):
        query = self.build()
        print(query)
        return self.model.fetchall(query)

    def raw(self, query):
        self.query = query
        return self

    def first(self):
        query = self.build()
        return self.model.fetchone(query)

    def insert(self, data):
        self.insert_data = data
        return self

    def update(self, data):
        self.update_data = data
        return self

    def delete(self):
        query = f'DELETE FROM {self.table}'
        if self.where_conditions:
            where_clause = " AND ".join(self.where_conditions)
            query += f' WHERE {where_clause}'
        self.delete_data = query
        return self

    def build(self):
        query = ''

        if self.insert_data:
            columns = ', '.join(self.insert_data.keys())
            values = ', '.join(str(value) for value in self.insert_data.values())
            query = f'INSERT INTO {self.table} ({columns}) VALUES ({values})'

        elif self.update_data:
            set_values = ', '.join(f'{key}={value}' for key, value in self.update_data.items())
            query = f'UPDATE {self.table} SET {set_values}'
            if self.where_conditions:
                where_clause = " AND ".join(self.where_conditions)
                query += f' WHERE {where_clause}'
                
        elif self.delete_data:
            return self.delete_data

        else:
            query = f'SELECT {", ".join(self.columns)} FROM {self.table}'

            if self.join_tables:
                for join_table, on_condition in self.join_tables:
                    query += f' INNER JOIN {join_table} ON {on_condition}'

            if self.where_conditions:
                where_clause = " ".join(self.where_conditions)
                query += f' WHERE {where_clause}'

            if self.group_by_columns:
                group_by_columns_str = ', '.join(self.group_by_columns)
                query += f" GROUP BY {group_by_columns_str}"

            if self.order_by_column:
                query += f' ORDER BY {self.order_by_column} {self.order_by_direction}'
        print(query)
        return query
