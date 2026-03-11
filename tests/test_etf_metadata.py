import unittest

from etf_data import ETFCodeMapper


class ETFMetadataTests(unittest.TestCase):
    def test_broker_etf_is_classified_as_industry(self):
        meta = ETFCodeMapper.get_etf_metadata("512880")
        self.assertEqual(meta["category"], "industry")
        self.assertEqual(meta["category_label"], "行业")
        self.assertEqual(meta["sector"], "financials")
        self.assertIn("broker", meta["tags"])
        self.assertTrue(ETFCodeMapper.is_industry("512880"))
        self.assertFalse(ETFCodeMapper.is_wide_basis("512880"))

    def test_hs300_etf_is_classified_as_wide_basis(self):
        meta = ETFCodeMapper.get_etf_metadata("510300")
        self.assertEqual(meta["category"], "wide_basis")
        self.assertEqual(meta["sector"], "broad_market")
        self.assertTrue(ETFCodeMapper.is_wide_basis("510300"))

    def test_consumer_name_can_be_inferred_as_industry(self):
        meta = ETFCodeMapper.get_etf_metadata("510150")
        self.assertEqual(meta["category"], "industry")
        self.assertEqual(meta["sector"], "consumer")


if __name__ == "__main__":
    unittest.main()
