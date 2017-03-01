class Matrix:
    def __init__(self, rows, cols):
        self.num_rows = rows
        self.num_cols = cols
        self.mat = [([0] * self.num_cols) for row_num in range(self.num_rows)]
        self.set_identity()

    def set_zero(self):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                self.mat[row][col] = 0
        return self

    def set_identity(self):
        self.set_zero()
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if row == col:
                    self.mat[row][col] = 1
        return self

    def all(self):
        for rows in self.mat:
            for val in rows:
                yield val

    def rows(self):
        for row in self.mat:
            yield row

    def columns(self):
        for col_num in range(self.num_cols):
            col = []
            for row in self.rows():
                col.append(row[col_num])
            yield col

    def col_size(self):
        return self.num_rows

    def row_size(self):
        return self.num_cols

    def dimension(self):
        return self.num_rows, self.num_cols

    """inner product, or matrix mul, depending of dimension"""
    def __mul__(self, other):
        if type(other) != Matrix:
            raise TypeError()
        if self.num_rows != other.num_cols:
            raise ValueError("dim incorrect: rows={0}, cols={1}".format(self.num_rows, other.num_cols))
        result = Matrix(self.num_rows, other.num_cols)
        for row_index, row in enumerate(self.rows()):
            for col_index, col in enumerate(other.columns()):
                result.mat[row_index][col_index] = sum([x * y for x, y in zip(row, col)])  # inner product
        return result

    def entrywise_product(self, other):
        return self.entrywise_operation(other, lambda a, b: a * b)

    def __sub__(self, other):
        return self.entrywise_operation(other, lambda a, b: a - b)

    def __add__(self, other):
        return self.entrywise_operation(other, lambda a, b: a + b)

    def entrywise_operation(self, other, operation):
        if type(other) != Matrix:
            raise TypeError()
        if self.dimension() != other.dimension():
            raise ValueError("dim mismatch")
        result = Matrix(self.num_rows, self.num_cols)
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                result.mat[row][col] = operation(self.mat[row][col], other.mat[row][col])
        return result

    def transpose(self):
        if self.num_rows != self.num_cols:
            raise ValueError("not square matrix")
        result = Matrix(self.num_rows, self.num_cols)
        for row in self.rows():
            for col in self.columns():
                result[col][row] = self.mat[row][col]

    def __getitem__(self, item):
        return self.mat[item[0]][item[1]]

    def __setitem__(self, key, value):
        self.mat[key[0]][key[1]] = value

    def __str__(self):
        output = ""
        for row in self.rows():
            output += str(row) + '\n'
        return output