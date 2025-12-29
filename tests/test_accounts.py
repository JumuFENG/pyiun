#!/usr/bin/env python3
import unittest
from unittest.mock import Mock, ANY, patch
import json
import sys
import os
import base64

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.accounts import accld
from app.stock_strategy import StockStrategyFactory as sfac


class TestAccounts(unittest.TestCase):
    def setUp(self):
        accld.dserver = 'http://localhost:9112/'
        accld.headers = {
            'Authorization': f'''Basic {base64.b64encode("chuazhou@outlook.com:123456".encode()).decode()}'''
        }

    def test_load_accounts(self):
        accld.load_accounts()
        self.assertGreater(len(accld.all_accounts), 0)

    def test_save_strategy(self):
        accld.load_accounts()
        account = accld.all_accounts.get('zt1bk')
        self.assertIsNotNone(account)
        code = 'sh600362'
        strategy_data = {
            'grptype': 'GroupStandard', 'strategies': {
                0: {'key': 'StrategySellELS', 'enabled': False, 'cutselltype': 'all', 'selltype': 'all', 'topprice': 4.55, 'guardPrice': 3.98},
                1: {'key': 'StrategySellBE', 'enabled': False, 'upRate': -0.03, 'selltype': 'all', 'sell_conds': 1}
            }, 'transfers': {
                0: {'transfer': '-1'}, 1: {'transfer': '-1'}
            },
            'amount': '5000'
        }
        account.save_stock_strategy(code, strategy_data)
        account.load_watchings()
        strategies = sfac.get_strategy_meta(account.keyword, code, 'StrategySellELS')
        self.assertIn('StrategySellELS', strategies['key'])


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestAccounts('test_save_strategy'))
    unittest.TextTestRunner().run(suite)
