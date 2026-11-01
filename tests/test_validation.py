import unittest
from src.data_loader import load_datasets
from src.data_validator import generate_statistics, infer_schema, validate_statistics

class TestValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_df, cls.eval_df, cls.serving_df, _ = load_datasets("data/raw")
        cls.stats = generate_statistics(cls.train_df)
        cls.schema = infer_schema(cls.stats)

    def test_schema_features(self):
        self.assertIn("annual_income", self.schema.features)
        self.assertIn("loan_intent", self.schema.features)

    def test_eval_anomalies(self):
        eval_stats = generate_statistics(self.eval_df)
        anom = validate_statistics(eval_stats, self.schema)
        self.assertTrue(len(anom) > 0)

if __name__ == '__main__':
    unittest.main()
