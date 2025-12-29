#!/usr/bin/env python3
"""
verify_strategies方法的单元测试 - 最终版本
"""
import unittest
from unittest.mock import Mock, ANY, patch
import json
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.accounts import Account, accld
from app.iuncld import iunCloud
from app.stock_strategy import StockStrategyFactory as sfac


class TestVerifyStrategies(unittest.TestCase):

    def setUp(self):
        """测试前的设置"""
        # 创建测试账户
        self.account = Account('test_account')

        # 模拟股票数据
        self.sample_stock_data = {
            "code": "603879",
            "name": "永悦科技",
            "holdCount": 700,
            "availableCount": 700,
            "strategies": {
                "grptype": "GroupStandard",
                "strategies": {
                    "1": {
                        "key": "StrategyBSBE",
                        "enabled": True,
                        "guardPrice": 7.79,
                        "stepRate": 0.08,
                        "selltype": "xsingle",
                        "disable_sell": False
                    },
                    "2": {
                        "key": "StrategySellELS",
                        "enabled": True,
                        "selltype": "all",
                        "cutselltype": "all",
                        "topprice": "7.75"
                    }
                },
                "transfers": {
                    "1": {"transfer": -1},
                    "2": {"transfer": -1}
                },
                "amount": 2000,
                "buydetail": [
                    {
                        "code": "603879",
                        "type": "B",
                        "price": 7.19,
                        "count": 200,
                        "date": "2025-08-25",
                        "sid": "754657"
                    }
                ],
                "buydetail_full": [
                    {
                        "code": "603879",
                        "type": "B",
                        "price": 7.19,
                        "count": 200,
                        "date": "2025-08-25",
                        "sid": "754657"
                    }
                ]
            },
            "holdCost": 7.786,
            "latestPrice": 7.18
        }

        # 模拟行情数据
        self.mock_quotes = {
            'price': 7.18,
            'high': 7.25,
            'low': 7.10,
            'open': 7.15,
            'lclose': 7.20,
            'top_price': 7.92,  # 涨停价
            'bottom_price': 6.48  # 跌停价
        }

    @patch('app.accounts.iunCloud')
    @patch('app.accounts.klPad')
    @patch('app.accounts.guang')
    @patch('app.accounts.logger')
    @patch('app.accounts.quotes')
    def test_verify_strategies_with_zt1bk_remark(self, mock_quotes_func, mock_logger, mock_guang, mock_klpad, mock_iuncloud):
        """测试有zt1bk交易备注的策略验证"""
        # 设置模拟返回值
        mock_guang.today_date.return_value = '2025-08-25'

        # 修改样本数据，设置高于昨收9%的价格以触发zt1bk策略
        high_quotes = self.mock_quotes.copy()
        high_quotes['high'] = 7.85  # 高于昨收7.20的9%


        self.sample_stock_data['strategies']['buydetail'] = [
            {"id": 1238,"code": "SH603879","date": "2025-08-04","count": 300,"price": 7.1,"sid": "422886","type": "B"},
            {"id": 1331,"code": "SH603879","date": "2025-08-22","count": 200,"price": 7.18,"sid": "1912045","type": "B"},
            {"code": "603879","type": "B","price": 7.19,"count": 200,"date": "2025-08-25","sid": "754657"}]
        self.sample_stock_data['strategies']['buydetail_full'] = [
            {"id": 1324,"code": "SH603879","date": "2025-08-01","count": 300,"price": 7.85,"sid": "391486","type": "B"},
            {"id": 1330,"code": "SH603879","date": "2025-08-04","count": 300,"price": 7.1,"sid": "422886","type": "B"},
            {"code": "603879","type": "B","price": 7.19,"count": 200,"date": "2025-08-25","sid": "754657"}
        ]

        mock_iuncloud.get_account_latest_stocks.return_value = [self.sample_stock_data]
        mock_quotes_func.return_value = {'603879': self.mock_quotes}
        mock_klpad.cache.return_value = None
        mock_klpad.get_quotes.return_value = high_quotes
        mock_klpad.get_zt_price.return_value = 7.92
        mock_guang.generate_strategy_json.return_value = {
            'strategies': {
                '0': {'key': 'StrategySellELS', 'enabled': True},
                '1': {'key': 'StrategySellBE', 'enabled': True}
            },
            'buydetail': [],
            'buydetail_full': []
        }

        # 设置账户的trading_remarks
        self.account.trading_remarks = {'603879': 'istrategy_zt1bk'}

        # 模拟save_stock_strategy方法
        self.account.save_stock_strategy = Mock()

        # 模拟get_strategy_meta方法
        sfac.get_strategy_meta = Mock(return_value=None)

        # 执行测试
        self.account.verify_strategies()

        # 验证结果
        mock_iuncloud.get_account_latest_stocks.assert_called_once_with('test_account')
        mock_quotes_func.assert_called_once_with(['603879'])
        mock_klpad.cache.assert_called_once()

        # 验证save_stock_strategy被调用并检查参数
        self.account.save_stock_strategy.assert_called()

        # 获取调用参数进行详细验证
        call_args = self.account.save_stock_strategy.call_args
        self.assertIsNotNone(call_args, "save_stock_strategy should have been called with arguments")

        args, kwargs = call_args
        # 验证第一个参数是股票代码
        self.assertTrue(len(args) >= 1, "save_stock_strategy should be called with at least 1 argument")
        self.assertEqual(args[0], '603879', "First argument should be stock code")

        # 验证第二个参数是策略数据
        if len(args) >= 2:
            strategy_data = args[1]
            self.assertIsInstance(strategy_data, dict, "Second argument should be a dictionary")
            self.assertIn('strategies', strategy_data, "Strategy data should contain 'strategies' key")

    @patch('app.accounts.iunCloud')
    @patch('app.accounts.klPad')
    @patch('app.accounts.guang')
    @patch('app.accounts.logger')
    def test_verify_strategies_no_holding(self, mock_logger, mock_guang, mock_klpad, mock_iuncloud):
        """测试无持仓股票的策略验证"""
        # 创建无持仓的股票数据
        no_holding_stock = self.sample_stock_data.copy()
        no_holding_stock['holdCount'] = 0

        mock_guang.today_date.return_value = '2025-08-25'
        mock_iuncloud.get_account_latest_stocks.return_value = [no_holding_stock]
        mock_klpad.get_quotes.return_value = self.mock_quotes

        # 模拟save_stock_strategy方法
        self.account.save_stock_strategy = Mock()

        # 执行测试
        self.account.verify_strategies()

        # 验证结果 - 应该调用verify_recycle_strategies
        self.account.save_stock_strategy.assert_called()

        # 验证调用参数
        call_args = self.account.save_stock_strategy.call_args
        if call_args:
            args, kwargs = call_args
            # 验证股票代码参数
            if args:
                self.assertEqual(args[0], '603879', "Stock code should be passed as first argument")

    def test_verify_zt_top_reached_with_zt_price(self):
        """测试涨停价格达到时的策略调整"""
        with patch('app.accounts.klPad') as mock_klpad, \
             patch('app.accounts.guang') as mock_guang:

            # 设置涨停价格
            zt_quotes = self.mock_quotes.copy()
            zt_quotes['price'] = 7.92  # 涨停价

            mock_klpad.get_quotes.return_value = zt_quotes
            mock_klpad.get_zt_price.return_value = 7.92
            mock_guang.generate_strategy_json.return_value = {
                'strategies': {
                    '0': {'key': 'StrategySellELS', 'enabled': True, 'topprice': 8.32, 'guardPrice': 7.52},
                    '1': {'key': 'StrategySellBE', 'enabled': True}
                }
            }

            # 模拟save_stock_strategy方法
            self.account.save_stock_strategy = Mock()

            # 执行测试
            result = self.account.verify_zt_top_reached(self.sample_stock_data)

            # 验证结果
            self.assertTrue(result)
            self.account.save_stock_strategy.assert_called_once()

            # 验证调用参数
            call_args = self.account.save_stock_strategy.call_args
            self.assertIsNotNone(call_args)
            args, kwargs = call_args
            # 验证传递了股票数据
            if args:
                stock_data = args[0] if args else None
                self.assertIsNotNone(stock_data, "Stock data should be passed to save_stock_strategy")

    @patch('app.accounts.iunCloud')
    @patch('app.accounts.klPad')
    @patch('app.accounts.guang')
    @patch('app.accounts.logger')
    @patch('app.accounts.quotes')
    def test_verify_cycling_strategies(self, mock_quotes_func, mock_logger, mock_guang, mock_klpad, mock_iuncloud):
        """测试有zt1bk交易备注的策略验证"""
        # 设置模拟返回值
        mock_guang.today_date.return_value = '2025-09-01'

        # 修改样本数据，设置高于昨收9%的价格以触发zt1bk策略
        mock_quotes = {
            'price': 14.64,
            'high': 14.85,
            'low': 14.23,
            'open': 14.24,
            'lclose': 14.10,
            'top_price': 16.10,  # 涨停价
            'bottom_price': 13.18  # 跌停价
        }

        code = '002125'
        self.sample_stock_data['code'] = code
        self.sample_stock_data['strategies'] = {
            "grptype": "GroupStandard",
            "strategies": {
                "0": {'key': 'StrategyBSBE', 'enabled': True, 'guardPrice': 19.93, 'selltype': 'xsingle', 'disable_sell': False, 'stepRate': 0.08},
                "1": {'key': 'StrategyBuyZTBoard', 'enabled': True}
            },
            "transfers": {
                "0": {"transfer": -1},
                "1": {"transfer": -1}
            },
            "amount": 2000,
        }
        self.sample_stock_data['strategies']['buydetail'] = [{'code': '002125', 'type': 'B', 'price': 14.64, 'count': 400, 'date': '2025-09-01', 'sid': '1968694'}, {'id': 905, 'code': 'SZ002125', 'date': '2025-06-26', 'count': 100, 'price': 16.84, 'sid': '74465', 'type': 'B'}]
        self.sample_stock_data['strategies']['buydetail_full'] = [{'code': '002125', 'type': 'B', 'price': 14.64, 'count': 400, 'date': '2025-09-01', 'sid': '1968694'}, {'id': 905, 'code': 'SZ002125', 'date': '2025-06-26', 'count': 100, 'price': 16.84, 'sid': '74465', 'type': 'B'}]

        mock_iuncloud.get_account_latest_stocks.return_value = [self.sample_stock_data]
        mock_quotes_func.return_value = {code: mock_quotes}
        mock_klpad.cache.return_value = None
        mock_klpad.get_quotes.return_value = mock_quotes
        mock_klpad.get_zt_price.return_value = 16.10
        mock_guang.generate_strategy_json.return_value = {
            'strategies': {
                '0': {'key': 'StrategyBSBE', 'enabled': True, 'guardPrice': 19.93, 'selltype': 'xsingle', 'disable_sell': False, 'stepRate': 0.08},
                '1': {'key': 'StrategyBuyZTBoard', 'enabled': True}
            },
            'buydetail': [],
            'buydetail_full': []
        }

        # 设置账户的trading_remarks
        self.account.keyword = 'normal'
        self.account.trading_remarks = {code: 'istrategy_hsrzt0'}

        # 模拟save_stock_strategy方法
        self.account.save_stock_strategy = Mock()

        # 模拟get_strategy_meta方法
        sfac.get_strategy_meta = Mock(return_value=None)

        # 执行测试
        self.account.verify_strategies()

        # 验证结果
        mock_iuncloud.get_account_latest_stocks.assert_called_once_with('normal')
        mock_quotes_func.assert_called_once_with([code])
        mock_klpad.cache.assert_called_once()

        self.account.save_stock_strategy.assert_called_with(code, ANY)

        # 验证调用参数
        call_args = self.account.save_stock_strategy.call_args
        self.assertIsNotNone(call_args, "save_stock_strategy should have been called")

        args, kwargs = call_args
        # 验证策略数据结构
        if len(args) >= 2 and isinstance(args[1], dict):
            strategy_data = args[1]
            self.assertIn('strategies', strategy_data, "Strategy data should contain strategies")

    def test_verify_strategies_class_method(self):
        """测试accld类的verify_strategies方法"""
        # 创建模拟账户
        mock_account1 = Mock()
        mock_account2 = Mock()

        # 直接设置accld.all_accounts
        original_accounts = accld.all_accounts
        accld.all_accounts = {
            'account1': mock_account1,
            'account2': mock_account2
        }

        try:
            # 执行测试
            accld.verify_strategies()

            # 验证所有账户的verify_strategies都被调用
            mock_account1.verify_strategies.assert_called_once()
            mock_account2.verify_strategies.assert_called_once()
        finally:
            # 恢复原始状态
            accld.all_accounts = original_accounts

    @patch('app.accounts.logger')
    def test_verify_strategies_with_exception(self, mock_logger):
        """测试verify_strategies方法异常处理"""
        # 创建会抛出异常的模拟账户
        mock_account = Mock()
        mock_account.verify_strategies.side_effect = Exception("Test exception")
        mock_account.keyword = 'test_account'

        # 直接设置accld.all_accounts
        original_accounts = accld.all_accounts
        accld.all_accounts = {'test_account': mock_account}

        try:
            # 执行测试
            accld.verify_strategies()

            # 验证异常被正确处理和记录
            mock_logger.error.assert_called()
        finally:
            # 恢复原始状态
            accld.all_accounts = original_accounts

    def test_get_strategy_meta(self):
        """测试获取策略元数据"""
        # 设置测试数据
        self.account.stocks = {
            '603879': {
                'strategies': {
                    'strategies': {
                        '1': {'key': 'StrategyBSBE', 'enabled': True, 'guardPrice': 7.79},
                        '2': {'key': 'StrategySellELS', 'enabled': True, 'topprice': 7.75}
                    }
                }
            }
        }

        # 测试获取存在的策略
        result = sfac.get_strategy_meta(self.account.keyword, '603879', 'StrategyBSBE')
        self.assertIsNotNone(result)
        self.assertEqual(result['key'], 'StrategyBSBE')
        self.assertEqual(result['guardPrice'], 7.79)

        # 测试获取不存在的策略
        result = sfac.get_strategy_meta(self.account.keyword, '603879', 'NonExistentStrategy')
        self.assertIsNone(result)

        # 测试获取不存在股票的策略
        result = sfac.get_strategy_meta(self.account.keyword, '000001', 'StrategyBSBE')
        self.assertIsNone(result)

    def test_save_stock_strategy_parameter_validation_examples(self):
        """演示验证save_stock_strategy调用参数的多种方法"""
        from unittest.mock import Mock, ANY, call

        # 模拟save_stock_strategy方法
        self.account.save_stock_strategy = Mock()

        # 模拟调用
        test_strategy_data = {
            'strategies': {'1': {'key': 'TestStrategy', 'enabled': True}},
            'buydetail': [],
            'amount': 1000
        }

        # 执行调用
        self.account.save_stock_strategy('603879', test_strategy_data)

        # 方法1: 验证是否被调用
        self.account.save_stock_strategy.assert_called()

        # 方法2: 验证调用次数
        self.account.save_stock_strategy.assert_called_once()

        # 方法3: 验证具体参数
        self.account.save_stock_strategy.assert_called_with('603879', test_strategy_data)

        # 方法4: 使用ANY匹配任意参数
        self.account.save_stock_strategy.assert_called_with('603879', ANY)

        # 方法5: 获取调用参数进行详细验证
        call_args = self.account.save_stock_strategy.call_args
        args, kwargs = call_args
        self.assertEqual(args[0], '603879')
        self.assertEqual(args[1]['amount'], 1000)
        self.assertIn('strategies', args[1])

        # 方法6: 验证所有调用记录
        all_calls = self.account.save_stock_strategy.call_args_list
        self.assertEqual(len(all_calls), 1)

        # 方法7: 使用call对象验证
        expected_call = call('603879', test_strategy_data)
        self.assertIn(expected_call, all_calls)

class TestIunCloudApis(unittest.TestCase):
    def test_getzdfranks(self):
        """测试iunCloud.getzdfranks方法"""
        # 调用方法
        result = iunCloud.get_stocks_zdfrank(8)
        # 验证结果类型
        self.assertIsInstance(result, list)

        # 验证涨跌幅是否符合条件
        for rank in result:
            self.assertIsInstance(rank, dict)


if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestSuite()
    # suite.addTest(TestVerifyStrategies('test_verify_cycling_strategies'))
    suite.addTest(TestIunCloudApis('test_getzdfranks'))
    unittest.TextTestRunner().run(suite)
