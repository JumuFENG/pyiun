import base64
import asyncio
import traceback
import stockrt as asrt
from app.guang import guang
from app.lofig import Config, logger, delayed_tasks
from app.trade_interface import TradeInterface
from app.accounts import accld
from app.klpad import klPad, DsvrKSource
from app.intrade_base import iunCloud
from app.strategy_factory import StrategyFactory, GlobalStartup


class iun:
    @classmethod
    async def intrade_matched(self, ikey, match_data, istr_message_creator):
        subscribe_detail = iunCloud.iun_str_conf(ikey)
        if subscribe_detail and callable(istr_message_creator):
            msg = istr_message_creator(match_data, subscribe_detail)
            if msg:
                TradeInterface.submit_trade(msg)
                logger.info(f'send {match_data}, {subscribe_detail}, {ikey}')

    @classmethod
    async def main(cls):
        remain_secs = guang.delay_seconds('9:11')
        if remain_secs > 0:
            logger.info(f'wait to run at 9:11')
            await asyncio.sleep(remain_secs)
        tconfig = Config.trading_service()
        TradeInterface.tserver = tconfig['server']
        if not TradeInterface.check_trade_server():
            logger.info('not trading day or trade server not available')
            return

        dconfig = Config.data_service()
        DsvrKSource.dserver = dconfig['server']
        iunCloud.dserver = dconfig['server']
        iunCloud.strFac = StrategyFactory
        accld.dserver = dconfig['server']
        accld.headers = {
            'Authorization': f'''Basic {base64.b64encode(f"{dconfig['user']}:{Config.simple_decrypt(dconfig['password'])}".encode()).decode()}'''
        }
        srtcfg = Config.stockrt_config()
        if 'default_sources' in srtcfg:
            for k, v in srtcfg['default_sources'].items():
                asrt.set_default_sources(k, v[0], tuple(v[1]), v[2])
        asrt.set_array_format(srtcfg.get('array_format', 'df'))
        accld.load_accounts()

        strategies = [GlobalStartup()] + [s for s in StrategyFactory.market_strategies() if s.key in TradeInterface.iun_str()]
        for task in strategies:
            task.on_intrade_matched = cls.intrade_matched
            await task.start_strategy_tasks()

        await asyncio.sleep(1)
        if len(delayed_tasks) > 0:
            # 处理延迟任务，包含异常处理
            results = await asyncio.gather(*delayed_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Delayed task {i} failed: {result}")
                    logger.error(traceback.format_exc())
            accld.verify_strategies()

        logger.info("iun main exited.")


if __name__ == '__main__':
    asyncio.run(iun.main())
