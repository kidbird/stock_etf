import types
import unittest
from unittest.mock import patch

import pandas as pd

from stock_data import StockCodeMapper


class StockDataTests(unittest.TestCase):
    def test_load_from_akshare_builds_live_industry_mapping(self):
        fake_ak = types.SimpleNamespace(
            stock_info_a_code_name=lambda: pd.DataFrame(
                {"code": ["000001", "300059"], "name": ["平安银行", "东方财富"]}
            ),
            stock_board_industry_name_em=lambda: pd.DataFrame(
                {"板块名称": ["银行", "证券"]}
            ),
            stock_board_industry_cons_em=lambda symbol: pd.DataFrame(
                {"代码": ["000001"] if symbol == "银行" else ["300059"]}
            ),
        )

        with patch.dict("sys.modules", {"akshare": fake_ak}):
            StockCodeMapper._live = {}
            StockCodeMapper._live_industry = {}
            count = StockCodeMapper.load_from_akshare()

        self.assertEqual(count, 2)
        self.assertEqual(StockCodeMapper.get_stock_industry("000001"), "银行")
        self.assertEqual(StockCodeMapper.get_stock_industry("300059"), "证券")


if __name__ == "__main__":
    unittest.main()
