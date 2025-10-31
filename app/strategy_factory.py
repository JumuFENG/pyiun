import asyncio
from functools import lru_cache
from app.lofig import logger, delayed_tasks
from app.market_strategy import *
from app.stock_strategy import *
from app.accounts import accld

class StrategyFactory():
    @classmethod
    def market_strategies(cls):
        return [
            StrategyI_AuctionUp(), StrategyI_Zt1Bk(), StrategyI_EndFundFlow(), StrategyI_DeepBigBuy(),
            StrategyI_3Bull_Breakup(), StrategyI_Zt1WbOpen(), StrategyI_HotrankOpen(), StrategyI_HotStocksOpen(),
            StrategyI_DtStocksUp(), StrategyI_HotstocksRetryZt0()
        ]

    @classmethod
    @lru_cache(maxsize=None)
    def stock_strategy(self, k, formkt=''):
        s = None
        if k == StrategyGE.key:
            s = StrategyGE()
        elif k == StrategyBuySellBeforeEnd.key:
            s = StrategyBuySellBeforeEnd()
        elif k == StrategySellELShort.key:
            s = StrategySellELShort()
        elif k == StrategySellBeforeEnd.key:
            s = StrategySellBeforeEnd()
        elif k == StrategySellMA.key:
            s = StrategySellMA()
        elif k == StrategyBuyZTBoard.key:
            s = StrategyBuyZTBoard()
        elif k == StrategyBuyDTBoard.key:
            s = StrategyBuyDTBoard()
        else:
            logger.error('Strategy not implemented: %s', k)

        if s:
            try:
                # 检查是否在asyncio上下文中
                loop = asyncio.get_running_loop()
                asyncio.create_task(s.start_strategy_tasks())
            except RuntimeError:
                # 不在asyncio上下文中，延迟到后续处理
                delayed_tasks.append(s.start_strategy_tasks())

        return s

    @classmethod
    def add_stock_strategy(cls, acc, code, strategy):
        if not isinstance(strategy, dict):
            return None

        accld.cache_stock_data(acc, code, {'strategies': strategy})
        for sobj in strategy['strategies'].values():
            if not sobj['enabled']:
                continue
            s = cls.stock_strategy(sobj['key'])
            if s:
                s.add_stock(acc, code)

    @classmethod
    def disable_stock_strategy(cls, acc, code, skey):
        if not skey:
            return

        smeta = accld.get_strategy_meta(acc, code, skey)
        if smeta and smeta['enabled']:
            smeta['enabled'] = False
            accld.update_strategy_meta(acc, code, skey, smeta)
        else:
            return

        s = cls.stock_strategy(smeta['key'])
        if s:
            s.remove_stock(acc, code)
        logger.info(f'stock  {acc} {code} {skey} disabled')
