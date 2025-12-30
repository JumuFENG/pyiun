#!/usr/bin/env python3
import unittest
from unittest.mock import Mock, ANY, patch
import json
import sys
import os
import base64

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.market_strategy import *
from app.iuncld import iunCloud


class TestMarketStrategy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        iunCloud.dserver = 'http://localhost:9112/'

        self._iun_str_conf_patcher = patch('app.iuncld.iunCloud.iun_str_conf')
        self.iun_str_conf = self._iun_str_conf_patcher.start()
        self.iun_str_conf.return_value ={'enabled': True, 'account': 'test', 'amount': 10000}

        self.submit_trade_patcher = patch('app.trade_interface.TradeInterface.submit_trade')
        self.submit_trade = self.submit_trade_patcher.start()
        self.submit_trade.return_value = True

    async def asyncTearDown(self):
        self._iun_str_conf_patcher.stop()
        self.submit_trade_patcher.stop()

    @unittest.skip('not implemented for StockAuctionUpSelector')
    async def test_StrategyI_AuctionUp(self):
        strategy = StrategyI_AuctionUp()
        await strategy.start_strategy_tasks()
        await strategy.watcher.check_dt_ranks()
        await strategy.watcher.notify_auctions1()
        await strategy.watcher.notify_auctions2()
        self.assertIsNotNone(strategy.matched)

    async def test_StrategyI_DtStocksUp(self):
        strategy = StrategyI_DtStocksUp()
        await strategy.start_strategy_tasks()
        await strategy.prepare_watcher.execute_task()
        self.assertIsInstance(strategy.candidates, dict)
        await strategy.watcher.execute_task()
        await strategy.watcher1.execute_task()
        await strategy.watcher2.execute_task()

    async def test_StrategyI_Zt1WbOpen(self):
        strategy = StrategyI_Zt1WbOpen()
        await strategy.start_strategy_tasks()
        await strategy.prepare_watcher.execute_task()
        self.assertIsInstance(strategy.candidates, dict)
        await strategy.watcher.execute_task()
        await strategy.watcher2.execute_task()

        self.assertTrue(self.submit_trade.called)

    async def test_StrategyI_HotrankOpen(self):
        strategy = StrategyI_HotrankOpen()
        await strategy.start_strategy_tasks()
        await strategy.prepare_watcher.execute_task()
        self.assertIsInstance(strategy.candidates, dict)
        await strategy.watcher.execute_task()
        await strategy.watcher2.execute_task()

    async def test_StrategyI_HotStocksOpen(self):
        strategy = StrategyI_HotStocksOpen()
        await strategy.start_strategy_tasks()
        await strategy.prepare_watcher.execute_task()
        self.assertIsInstance(strategy.candidates, dict)
        await strategy.watcher.execute_task()

    async def test_StrategyI_3Bull_Breakup(self):
        strategy = StrategyI_3Bull_Breakup()
        await strategy.start_strategy_tasks()
        await strategy.prepare_watcher.execute_task()
        self.assertIsInstance(strategy.candidates, dict)
        await strategy.watcher.execute_task()

    async def test_StrategyI_Zt1Bk(self):
        strategy = StrategyI_Zt1Bk()
        await strategy.start_strategy_tasks()
        await strategy.bkwatcher.execute_task()
        await strategy.watcher.execute_task()

    @unittest.skip("need to get fund flow history data")
    async def test_StrategyI_EndFundFlow(self):
        strategy = StrategyI_EndFundFlow()
        asrt.set_array_format('df')
        await strategy.start_strategy_tasks()
        await strategy.watcher.execute_task()

    async def test_StrategyI_DeepBigBuy(self):
        strategy = StrategyI_DeepBigBuy()
        await strategy.start_strategy_tasks()
        await strategy.watcher.execute_task()

    async def test_StrategyI_HotstocksRetryZt0(self):
        asrt.set_array_format('df')
        strategy = StrategyI_HotstocksRetryZt0()
        await strategy.start_strategy_tasks()
        await strategy.watcher.execute_task()


if __name__ == '__main__':
    # unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(TestMarketStrategy('test_StrategyI_AuctionUp'))
    unittest.TextTestRunner().run(suite)
