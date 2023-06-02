from numpy.random import rand

class Imputer:

    def __init__(self):
        pass

    def fill_dummy_values(self, scaling_factor):

        dummy_data = self.data.copy()

        for col in dummy_data:

            # Get column, column missing values and range
            col = dummy_data[col]
            col_null = col.isnull()
            num_nulls = col_null.sum()
            col_range = col.max() - col.min()

            # Shift and scale dummy values
            dummy_values = (rand(num_nulls) - 2)
            dummy_values = dummy_values * scaling_factor * col_range + col.min()

            # Return dummy values
            col[col_null] = dummy_values

        return dummy_data