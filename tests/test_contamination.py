import numpy as np
import pandas as pd
from proteomics_downstream_analysis.contamination import ContaminationAnalysis
import unittest
from pytest import approx

# pylint: disable=protected-access
# pylint: disable=missing-function-docstring


class TestContamination(unittest.TestCase):

    def setUp(self):
        
        self.helper = ContaminationAnalysis()

        # create test data 1
        self.data = pd.DataFrame({
            "a": list("abcdefg"), 
            "b": [1, 2, 3, 4, 5, 6, 7],
            "c": [2, 2, 9, 5, 8, 7, 4],
            "d": [1, 10, 8, 4, 5, 6, 7],
            "e": [100, 8218, 8959, 4833, 485, 45, 485]
        })
        self.data.columns = ["Genes","a", "a", "a", "a"]
        self.data["a"] = self.data["a"].astype(float)
        self.data["Genes"] = self.data["Genes"].astype("string")

        # create test data 2
        self.data2 = pd.DataFrame({"Genes" : np.array(["PRDX2", "CA2", "CAT", "d", "e", "f", "g"]),
                                    "b" : np.array([1, 2, 3, 4, 5, 6, 7]),
                                    "c" : np.array([2, 2, 9, 5, 8, 7, 4]),
                                    "e" : np.array([1, 21, 8, 4, 5, 6, 7]),
                                    "f" : np.array([1, 10, 8, 4, 5, 6, 7]),
                                    "g" : np.array([1, 10, 8, 4, 5, 6, 7]),
                                    "h" : np.array([1, 10, 8, 4, 5, 6, 7]),
                                    "i" : np.array([1, 10, 8, 4, 5, 6, 7]),
                                    "j" : np.array([412, 231, 234, 5, 1, 3, 4]),
                                    }
                        )

        self.data2.columns = ["Genes","a", "a","a", "a","a", "a","a", "a"]
        self.data2["a"] = self.data2["a"].astype(float)
        self.data2["Genes"] = self.data2["Genes"].astype("string")

        # create test data 3
        self.data3 = pd.DataFrame({"a" : np.array(["a", "b", "c", "d", "e", "f", "g"]),
                                "b" : np.array([1, np.nan, 3, 4, 5, 6, 7]),
                                "c" : np.array([2, 2, np.nan, np.nan, np.nan, np.nan, np.nan]),
                                "e" : np.array([np.nan, np.nan, 3, 4, 5, 6, 7]),
                                "f" : np.array([1, 10, 8, 4, 5, 6, 7]),
                                "g" : np.array([1, 10, 8, 4, 5, 6, 7]),
                                "h" : np.array([1, 10, 8, 4, 5, 6, 7]),
                                "i" : np.array([1, 10, 8, 4, 5, 6, 7]),
                                "j" : np.array([3, 24, 64, 42, 23, 5, 32]),}
                                )

        self.data3.columns = ["Genes", "a", "a", "a", "a", "a", "a", "a", "a"]
        self.data3["a"] = self.data3["a"].astype("float")
        self.data3["Genes"] = self.data3["Genes"].astype("string")

        self.data4 = pd.DataFrame({"Genes": ["a", "b", "c"],
                      "a": [2.0, 5.0, 1.0],
                      "c": [3.0, 4.0, 2.0], 
                      "b": [4.0, 3.0, 3.0],})
        self.data4["Genes"] = self.data4["Genes"].astype("string")

        # create test data 3
        self.data5 = pd.DataFrame({"a" : np.array(["a", "b", "c",]),
                                "b" : np.array([1, 1, 1]),
                                "c" : np.array([2, 2, 2]),
                                "e" : np.array([3, 3, 3]),
                                "f" : np.array([4, 4, 4]),
                                "g" : np.array([5, 5, 5]),
                                "h" : np.array([6, 6, 6]),
                                "i" : np.array([7, 7, 7]),
                                "j" : np.array([8, 8, 8]),}
                                )

        self.data5.columns = ["Genes", "a", "b", "a", "b", "a", "b", "a", "b"]
        self.data5["a"] = self.data5["a"].astype("float")
        self.data5["b"] = self.data5["b"].astype("float")
        self.data5["Genes"] = self.data5["Genes"].astype("string")

    def test_sort_by_column_names(self):
        output = self.helper._sort_by_column_names(data=self.data5)
        expected_result = ["a", "a", "a", "a", "b", "b", "b", "b"]
        self.assertTrue(output.columns.tolist() == expected_result)

    def test_missing_values_removal(self):
        output = self.helper.outlier(data=self.data3, kind="missing values", remove=True)
        expected_result  = 7
        self.assertTrue(output.shape[1] == expected_result)

    def test_missing_values_no_removal(self):
        output = self.helper.outlier(data=self.data3, kind="missing values")
        self.assertTrue((output[0] == np.array([0, 2, 3, 4, 5, 6, 7])).all())
        
    def test_contamination_removal(self):

        output = self.helper.outlier(self.data2, "contamination", True)
        expected_result = 7

        self.assertTrue(output.shape[1] == expected_result)

    def test_contamination_no_removal(self):
        output = self.helper.outlier(self.data2, "contamination", False)
        expected_value = np.array([0, 1, 2, 3, 4, 5, 6])
        self.assertTrue((output[0] == expected_value).all())
        
    def test_outlier_zscore_removal(self):
         
        output = self.helper.outlier(self.data, remove=True)
        expected_value = 4
        self.assertTrue(output.shape[1] == expected_value)

    def test_outlier_zscore_no_removal(self):
        
        output = self.helper.outlier(self.data)
        expected_result = np.array([0, 1, 2])
        self.assertTrue((output[0] == expected_result).all())

    def test_count_missing_values(self):
         
        nan_values = self.helper._count_missing_values(self.data3)
        expected_values = np.array([1, 5, 2, 0, 0, 0, 0, 0])

        self.assertTrue((nan_values == expected_values).all()) 

    def test_compute_rbc_total_ratio(self):
         
        rbc_total_ratio = self.helper._compute_rbc_total_ratio(self.data2)
        
        assert rbc_total_ratio[1] == approx(13/37)

    def test_outliers(self):
         mask = self.helper._get_outlier_mask(5, np.arange(10))
         self.assertTrue((mask == np.array([False, False, False, False, False, False, True, True, True, True])).all())

    def test_upper_limit(self):
            upper_lim = self.helper._upper_limit(np.arange(10))
            assert upper_lim == approx(20.25)

    def test_number_of_protein_outliers(self):
        outliers = self.helper._number_of_protein_outliers(self.data4.select_dtypes(float))
        self.assertTrue(outliers[0] == 1)

    def test_robust_zscore(self):
        zscores = self.helper._robust_zscore(self.data)
        assert zscores.iloc[0, 0] == approx((0.6745 * (2.0 - 3.0)/1))

if __name__ == "__main__":
    unittest.main()