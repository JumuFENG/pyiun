import base64
import asyncio
import traceback
import stockrt as asrt
from app.guang import guang
from app.lofig import Config, logger, delayed_tasks
from app.trade_interface import TradeInterface
from app.accounts import accld
from app.klpad import DsvrKSource
from app.iuncld import iunCloud
from app.market_strategy import MarketStrategyFactory as mfac


class iun:
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

        await mfac.start_all()

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
