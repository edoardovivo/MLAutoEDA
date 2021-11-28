import unittest
import math
import pandas as pd
import numpy as np
import os
from MLAutoEDA import MLAutoEDA


class MLAutoEDATestCase(unittest.TestCase):

    def setUp(self):
        DATA_PATH = os.path.join(os.getcwd(), 'titanic')
        train_df = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
        #test_df = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))
 
        train_df["Survived"] = train_df["Survived"].astype("category")
        
        dataframe = train_df
        numerical_vars = ['Age', 'SibSp', 'Fare']
        categorical_vars = ['Pclass', 'Sex', 'Ticket', 'Cabin', 'Embarked']
        dataframe[categorical_vars] = dataframe[categorical_vars].astype("category")
        for c in categorical_vars:
            if ("Unknown" not in dataframe[c].cat.categories):
                dataframe.loc[:, c] = dataframe[c].cat.add_categories("Unknown").fillna("Unknown")
        target = 'Survived'
        target_type = 'categorical'
        
        
        self.autoEDA = MLAutoEDA(dataframe, numerical_vars, categorical_vars, target, target_type)

    def test_entropy(self):
        """Test entropy"""

        # 0 multiplied by 2 return 0
        result = self.autoEDA.entropy("Pclass")
        self.assertEqual(result, 0.9976616191577425)

    def test_conditional_entropy(self):
        """Test conditional entropy"""

        # 5 multiplied by 2 return 10
        result = self.autoEDA.conditional_entropy("Pclass", "Survived")
        self.assertEqual(result, 0.9395543664674189)

    def test_eta(self):
        """Test correlation ratio eta"""

        # -7 multiplied by 2 return -14
        result = self.autoEDA.eta("Pclass","Age")
        self.assertEqual(result, 0.4152296884337575)

    def test_theil_U(self):
        """Test Theil's U coefficient"""
        
        result = self.autoEDA.theil_U( "Pclass","Survived")
        self.assertEqual(result, 0.05824344805343886)

    def test_compute_association_num_vs_num(self):
        """Test computing association numerical vs numerical"""

        result = self.autoEDA.compute_association( "Age", "numerical", "Fare", "numerical")
        self.assertEqual(result, 0.0960666917690389)
    
    
    def test_compute_association_num_vs_cat(self):
        """Test computing association numerical vs categorical"""

        result = self.autoEDA.compute_association( "Age", "numerical", "Pclass", "categorical")
        self.assertEqual(result, 0.4152296884337575)
    
    
    def test_compute_association_cat_vs_cat(self):
        """Test computing association categorical vs categorical"""

        result = self.autoEDA.compute_association( "Pclass", "categorical", "Survived", "categorical")
        self.assertEqual(result, 0.05824344805343886)
    
    


if __name__ == '__main__':
    unittest.main()
