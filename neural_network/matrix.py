import random
import math
import copy


class Matrix:
    def __init__(self, rows, cols):
        self.num_rows = rows
        self.num_cols = cols
        self.mat = [([0] * self.num_cols) for row_num in range(self.num_rows)]
        self.set_identity()

    def copy(self):
        new_mat = Matrix(0, 0)
        new_mat.num_rows = self.num_rows
        new_mat.num_cols = self.num_cols
        new_mat.mat = copy.deepcopy(self.mat)
        return new_mat

    def set_each_entry(self, operator):
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                self.mat[row][col] = operator(self.mat[row][col], row, col)

    def set_zero(self):
        self.set_each_entry(lambda value, row, col: 0)

    def set_identity(self):
        self.set_each_entry(lambda value, row, col: 1 if row == col else 0)
        return self

    def set_randomize(self):
        self.set_each_entry(lambda value, row, col: random.uniform(0.0, 1.0))
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

    """inner product, or matrix mul, depending of dimension. Expects a matrix"""
    def __mul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError()
        if self.row_size() != other.col_size():
            raise ValueError("dim incorrect: rows={0}, cols={1}".format(self.num_rows, other.num_cols))
        result = Matrix(self.num_rows, other.num_cols)
        for row_index, row in enumerate(self.rows()):
            for col_index, col in enumerate(other.columns()):
                result.mat[row_index][col_index] = sum([x * y for x, y in zip(row, col)])  # inner product
        return result

    def entrywise_product(self, other):
        return self.entrywise_operation(other, lambda a, b: a * b)

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            copy_mat = self.copy()
            copy_mat.set_each_entry(lambda value, row, col: value - other)
            return copy_mat
        return self.entrywise_operation(other, lambda a, b: a - b)

    def __rsub__(self, other):
        raise NotImplemented()

    def __add__(self, other):
        if not isinstance(other, Matrix):
            copy_mat = self.copy()
            copy_mat.set_each_entry(lambda value, row, col: value + other)
            return copy_mat
        return self.entrywise_operation(other, lambda a, b: a + b)

    def __radd__(self, other):
        copy_mat = self.copy()
        copy_mat.set_each_entry(lambda value, row, col: value + other)
        return copy_mat

    def __truediv__(self, other):
        if isinstance(other, Matrix):
            raise NotImplemented()
        copy_mat = self.copy()
        copy_mat.set_each_entry(lambda value, row, col: value / other)
        return copy_mat

    def __rtruediv__(self, other):
        if isinstance(other, Matrix):
            raise NotImplemented()
        copy_mat = self.copy()
        copy_mat.set_each_entry(lambda value, row, col: other / value)
        return copy_mat

    def __neg__(self):
        copy_mat = self.copy()
        copy_mat.set_each_entry(lambda value, row, col: -value)
        return copy_mat

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
        result = Matrix(self.num_cols, self.num_rows)
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                result[col, row] = self[row, col]
        return result

    def set_transpose(self):
        self.mat = self.transpose().mat
        return self

    def __getitem__(self, item):
        return self.mat[item[0]][item[1]]

    def __setitem__(self, key, value):
        self.mat[key[0]][key[1]] = value

    def __str__(self):
        output = ""
        for row in self.rows():
            output += str(row) + '\n'
        return output[:-1]  # I'm lazy, sue me


class Vector(Matrix):
    def __init__(self, length):
        self.length = length
        super().__init__(length, 1)

    def __mul__(self, other):
        if type(other) != Vector:
            return super().__mul__(other)
        if len(other) != len(self):
            raise ValueError("length mismatch")
        return super().__mul__(other.transpose())[0, 0]

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return super().__getitem__((item, 0))

    def __setitem__(self, key, value):
        super().__setitem__((key, 0), value)

    def as_list(self):
        return self.transpose()


def exp(m):
    result = m.copy()
    result.set_each_entry(lambda value, row, col: math.exp(value))
    return result
