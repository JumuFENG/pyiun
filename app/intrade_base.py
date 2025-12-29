import traceback
import json
import requests
import concurrent, concurrent.futures
from functools import lru_cache
from typing import Dict, List
from urllib.parse import urlencode
import stockrt as asrt
from app.lofig import logger, Config
from app.guang import guang
from app.klpad import klPad
from app.watcher_base import *
from app.iuncld import iunCloud


class BaseStrategy:
    def __init__(self):
        self.watcher: Watcher_Once = None

    async def start_strategy_tasks(self):
        assert hasattr(self, 'watcher'), 'watcher not set!'
        self.watcher.add_listener(self)
        await self.watcher.start_strategy_tasks()

    async def on_watcher(self, params):
        pass

    def on_taskstop(self):
        pass


class StockStrategy(BaseStrategy):
    def __init__(self):
        self.accstocks = []
        self.watchers = []

    def add_stock(self, acc, code):
        if (acc, code) not in self.accstocks:
            self.accstocks.append((acc, code))
        for w in self.watchers:
            w.add_stock(code)

    def remove_stock(self, acc, code, watcher=None):
        if (acc, code) in self.accstocks:
            self.accstocks.remove((acc, code))
        if watcher is not None:
            watcher.remove_stock(code)
        else:
            for w in self.watchers:
                w.remove_stock(code)

    async def start_strategy_tasks(self):
        for w in self.watchers:
            w.add_listener(self)
            await w.start_strategy_tasks()

    async def on_watcher(self, params):
        for code in params:
            kltypes = params[code]
            if not kltypes:
                continue
            for acc, acode in self.accstocks:
                if code == acode:
                    await self.check_kline(acc, code, kltypes)

    async def check_kline(self, acc, code, kltypes):
        # 这里可以添加K线检查的代码
        pass


class AuctionSnapshot_Watcher(Watcher_Cycle):
    auction_quote = {}
    def __init__(self):
        super().__init__(5, '9:20:2', '9:25:8')
        w2 = Watcher_Once('9:24:53')
        w2.execute_task = self.notify_auctions1
        w3 = Watcher_Once('9:25:16')
        w3.execute_task = self.notify_auctions2
        w1 = Watcher_Once('9:20:1', w3.btime)
        w1.execute_task = self.check_dt_ranks
        self.simple_watchers = [w1, w2, w3]
        self.matched = []

    async def check_dt_ranks(self):
        logger.info('check_dt_ranks')
        res = guang.get_request('http://33.push2.eastmoney.com/api/qt/clist/get', params={
            'pn': 1,
            'pz': 100,
            'po': 0,
            'np': 1,
            'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
            'fltt': 2,
            'invt': 2,
            'wbp2u': '|0|0|0|web',
            'fid': 'f3',
            'fs': 'm:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048',
            'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f115,f152'
        }, headers=guang.em_headers(Host='33.push2.eastmoney.com'))
        if res is None:
            return

        r = json.loads(res)
        if r['data'] is None or len(r['data']['diff']) == 0:
            return

        dtcodes = []
        for rkobj in r['data']['diff']:
            c = rkobj['f2']   # 最新价
            zd = rkobj['f3']  # 涨跌幅
            ze = rkobj['f4']  # 涨跌额
            if c == '-' or zd == '-' or ze == '-' or zd > -8:
                continue
            cd = rkobj['f12'] # 代码
            lclose = rkobj['f18']
            topprc = guang.zt_priceby(lclose, zdf=guang.zdf_from_code(cd))
            bottomprc = guang.dt_priceby(lclose, zdf=guang.zdf_from_code(cd))
            m = rkobj['f13']  # 市场代码 0 深 1 沪
            self.auction_quote[cd] = {
                'quotes': self.get_trends(f'{m}.{cd}'), 'lclose': lclose,
                'topprice': topprc, 'bottomprice': bottomprc}
            dtcodes.append(cd)

        url = guang.join_url(iunCloud.dserver, 'stock') + '?act=zdtindays&codes=' + ','.join(dtcodes) + '&date=' + guang.today_date('-')
        zddaysdt = json.loads(guang.get_request(url))
        for code, zddays in zddaysdt.items():
            code = code[-6:]
            if not self.auction_quote[code]:
                continue
            self.auction_quote[code]['zddays'] = zddays

    def get_trends(self, secid):
        trends_data = guang.get_request('http://push2his.eastmoney.com/api/qt/stock/trends2/get', params={
            'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58',
            'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
            'ndays': 1,
            'iscr': 1,
            'iscca': 0,
            'secid': secid
        }, headers=guang.em_headers(Host='push2his.eastmoney.com'))

        trends_data = json.loads(trends_data)
        trends = []
        if 'data' in trends_data and 'trends' in trends_data['data']:
            for trd in trends_data['data']['trends']:
                trds = trd.split(',')
                ttime = trds[0].split()[1]
                trends.append([ttime, float(trds[1]), 0, 0])
                if trds[2] != trds[1]:
                    trends.append([ttime+':01', float(trds[2]), 0, 0])

        return trends[1:]

    async def execute_task(self):
        quotes = asrt.quotes5(list(self.auction_quote.keys()))
        self.cache_quotes(quotes)

    def cache_quotes(self, quotes):
        for code, quote in quotes.items():
            try:
                price = quote['price']
                if quote['open'] == 0 and quote['bid1'] == quote['ask1']:
                    price = quote['bid1']
                    matched_vol = quote['bid1_volume']
                    buy2_count = quote['bid2_volume']
                    sell2_count = quote['ask2_volume']
                    unmatched_vol = buy2_count if buy2_count > 0 else -sell2_count
                else:
                    matched_vol = quote['volume']
                    unmatched_vol = 0
                    if quote['price'] == quote['bid1']:
                        unmatched_vol = quote['bid1_volume']
                    elif quote['price'] == quote['ask1']:
                        unmatched_vol = -quote['ask1_volume']
                self.auction_quote[code]['quotes'].append([quote['time'], price, matched_vol, unmatched_vol])
            except Exception as e:
                logger.error('exception in cache autcion quotes: %s, %s', code, quote)
                logger.error(traceback.format_exc())

    async def notify_auctions1(self):
        await self.notify_change({'quotes': self.auction_quote, 'uppercent': 5})

    async def notify_auctions2(self):
        await self.notify_change({'quotes': self.auction_quote, 'uppercent': 2})

    async def stop_simple_task(self, delay=0):
        await super().stop_simple_task(delay)
        if iunCloud.save_db_enabled():
            aucurl = guang.join_url(iunCloud.dserver, 'stock')
            today = guang.today_date('-')
            guang.post_data(aucurl, data={'act': 'save_auction_details', 'date': today, 'auctions': json.dumps(self.auction_quote)})

            values = []
            for c in self.auction_quote:
                if c not in self.matched:
                    continue
                q = self.auction_quote[c]
                if q['topprice'] == '-' and q['bottomprice'] == '-':
                    continue
                zdays, zdist, ddays, ddist = q['zddays']
                values.append([c, today, q['topprice'], q['bottomprice'], zdays, zdist, ddays, ddist])

            aucmatchurl = guang.join_url(iunCloud.dserver, 'stock')
            guang.post_data(aucmatchurl, data={'act': 'save_auction_matched', 'matched': json.dumps(values)})


class StkZdfJobProcess(JobProcess):
    def __init__(self, task_queue, result_queue, period):
        super().__init__(task_queue, result_queue, period)
        self.min_zdf = 8

    def process_prepare(self):
        asrt.set_default_sources('stock_list', 'stocklistapi', ('sina', 'cls', 'tencent', 'xueqiu'), False)

    def process_job(self, indata):
        full_zdf = []
        zdfranks = iunCloud.get_stocks_zdfrank(self.min_zdf)
        for rkobj in zdfranks:
            c = rkobj['close']   # 最新价
            zd = rkobj['change'] * 100  # 涨跌幅
            if c == '-' or zd == '-':
                continue
            if zd < self.min_zdf:
                break
            code = rkobj['code'][-6:] # 代码
            lc = rkobj['lclose'] # 昨收
            full_zdf.append([code, zd, c, lc])
        logger.info('StkZdfJobProcess %s stocks > %d', len(full_zdf), self.min_zdf)
        return full_zdf


class StkZdf_Watcher(SubProcess_Watcher_Cycle):
    ''' 个股涨幅排行,仅获取涨跌幅>=8%
    '''
    def __init__(self):
        super().__init__(60, '9:30:1', '14:57:1', [('11:30:1', '13:00:1')])
        self.full_zdf = []

    def create_subprocess(self):
        return StkZdfJobProcess(self.task_queue, self.result_queue, self.period)

    async def handle_process_result(self, result):
        if not result:
            logger.warning("get stock zdf empty")
            return

        try:
            await self.notify_change(result)
            self.full_zdf = result
        except Exception as e:
            logger.error(f"Error notify stkzdf: {e}")
            logger.error(traceback.format_exc())


class StkChgsJobProcess(JobProcess):
    def __init__(self, task_queue, result_queue, period):
        super().__init__(task_queue, result_queue, period)
        self.exist_changes = set()
        self.chg_pagesize = 1000
        self.session = None

    def process_job(self, indata):
        self.chg_page = 0
        self.fecthed = []
        self.get_next_changes()
        self.fecthed.reverse()
        if 0 < len(self.fecthed) < self.chg_pagesize:
            logger.info(f'fecthed {len(self.fecthed)}')
            self.chg_pagesize = max(64, len(self.fecthed))
        return self.fecthed

    def get_next_changes(self, types=None):
        if types is None:
            types = '8213,8201,8193,8194,64,128,4,16'
        # 60日新高,火箭发射, 大笔买入, 大笔卖出, 有大买盘, 有大卖盘, 封涨停板, 打开涨停板
        url = f'https://push2ex.eastmoney.com/getAllStockChanges?type={types}&cb=&ut=7eea3edcaed734bea9cbfc24409ed989&pageindex={self.chg_page}&pagesize={self.chg_pagesize}&dpt=wzchanges&_={guang.time_stamp()}'
        if self.session is None:
            self.session = requests.Session()
            self.session.timeout = 5
            headers = guang.em_headers(Host='push2ex.eastmoney.com', Referer='https://quote.eastmoney.com/changes/')
            headers['Accept-Encoding'] = 'gzip, deflate, br, zstd'
            headers['Priority'] = 'u=4'
            self.session.headers.update(headers)
        rsp = self.session.get(url)
        chgs = rsp.json()
        if 'data' not in chgs or chgs['data'] is None:
            return

        if 'allstock' in chgs['data']:
            self.merge_fetched(chgs['data']['allstock'])

        if len(self.fecthed) + len(self.exist_changes) < chgs['data']['tc']:
            self.chg_page += 1
            self.get_next_changes(types)

    def merge_fetched(self, changes):
        f2ch = ['00', '60', '30', '68', '83', '87', '43', '92', '90', '20']
        for chg in changes:
            code = chg['c']
            if code[0:2] not in f2ch:
                logger.warning(f'unknown code {chg}')
                continue
            tm = str(chg['tm']).rjust(6, '0')
            ftm = f'{tm[0:2]}:{tm[2:4]}:{tm[4:6]}'
            tp = chg['t']
            info = chg['i']
            if (code, ftm, tp) not in self.exist_changes:
                self.fecthed.append([code, ftm, tp, info])
                self.exist_changes.add((code, ftm, tp))


class StkChanges_Watcher(SubProcess_Watcher_Cycle):
    ''' 盘中异动
    '''
    def __init__(self):
        super().__init__(60, '9:30:1', '14:57:1', [('11:30:1','13:00:1')])

    def create_subprocess(self):
        return StkChgsJobProcess(self.task_queue, self.result_queue, self.period)

    async def handle_process_result(self, result):
        if not result:
            logger.warning("get stock changes empty")
            return

        try:
            await self.notify_change(result)
        except Exception as e:
            logger.error(f"Error notify bk changes: {e}")
            logger.error(traceback.format_exc())


class BkChgsJobProcess(JobProcess):
    def __init__(self, task_queue, result_queue, period):
        super().__init__(task_queue, result_queue, period)

    def process_job(self, bkchgurl):
        params = {
            'act': 'rtbkchanges',
            'save': 1
        }
        rsp = guang.get_request(bkchgurl, params=params)
        return json.loads(rsp)


class BKChanges_Watcher(SubProcess_Watcher_Cycle):
    def __init__(self):
        super().__init__(600, '9:30:45', '15:1:5', [('11:30', '12:50:45')])

    def feed_process_data(self):
        self.task_queue.put(guang.join_url(iunCloud.dserver, 'stock'))

    def create_subprocess(self):
        return BkChgsJobProcess(self.task_queue, self.result_queue, self.period)

    async def handle_process_result(self, result):
        if not result:
            logger.warning("get bk changes empty")
            return

        try:
            await self.notify_change(result)
        except Exception as e:
            logger.error(f"Error notify bk changes: {e}")
            logger.error(traceback.format_exc())


class EndFundFlow_Watcher(Watcher_Once):
    ''' 主力资金流
    '''
    def __init__(self):
        super().__init__('14:57:55')

    def updateLatestFflow(self):
        """获取最新主力资金流数据"""
        DEFAULT_PAGE_SIZE = 100
        BASE_URL = 'https://push2.eastmoney.com/api/qt/clist/get'
        COMMON_PARAMS = {
            'fid': 'f62',
            'po': 1,
            'np': 1,
            'fltt': 2,
            'invt': 2,
            'ut': 'b2884a393a59ad64002292a3e90d46a5'
        }
        FIELDS = 'fields=f1,f2,f3,f12,f13,f14,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f124'
        FS = 'fs=m:0+t:6+f:!2,m:0+t:13+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2'
        date = guang.today_date('-')
        headers = guang.em_headers(Host='push2.eastmoney.com', Referer='https://data.eastmoney.com/zjlx/detail.html')
        mainflows = []

        def build_url(pageno):
            params = {
                **COMMON_PARAMS,
                'pz': DEFAULT_PAGE_SIZE,
                'pn': pageno
            }
            return f"{BASE_URL}?{urlencode(params)}&{FS}&{FIELDS}"

        def process_response(response):
            """处理API响应数据"""
            try:
                data = json.loads(response)
                if data.get('data') and data['data'].get('diff'):
                    return data['data']['diff'], data['data'].get('total', 0)
                return [], 0
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing response: {e}")
                return [], 0

        def add_mainflow(fdatadiff):
            """添加有效的主力资金流数据"""
            for fobj in fdatadiff:
                if fobj.get('f62') == '-' or fobj.get('f184') == '-':
                    continue
                secid = f"{fobj['f13']}.{fobj['f12']}"
                mainflows.append([secid, date, fobj['f62'], fobj['f184']])

        # 获取第一页数据并确定总页数
        first_page_diff, total = process_response(guang.get_request(build_url(1), headers=headers))
        if not first_page_diff:
            return mainflows

        add_mainflow(first_page_diff)

        # 计算总页数 (修正点)
        total_pages = max(1, (total + DEFAULT_PAGE_SIZE - 1) // DEFAULT_PAGE_SIZE)
        if total_pages <= 1:
            return mainflows

        # 使用线程池并发获取剩余页面
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(
                    lambda p: add_mainflow(process_response(guang.get_request(build_url(p), headers))[0]),
                    pageno
                ): pageno
                for pageno in range(2, total_pages + 1)
            }

            for future in concurrent.futures.as_completed(futures):
                pageno = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing page {pageno}: {e}")

        return mainflows

    async def execute_task(self):
        mflow = self.updateLatestFflow()
        await self.notify_change(mflow)


class Stock_KlineDay_Watcher(Watcher_Once, Stock_Rt_Watcher):
    def __init__(self):
        super().__init__('14:57:55')
        Stock_Rt_Watcher.__init__(self)

    async def execute_task(self):
        codes = [c for c in self.codes if self.codes[c] > 0]
        try:
            chgklt = klPad.load_dsvr_klines(codes, kltype=101, length=32)
        except Exception as e:
            logger.error(f"Error get daily kline data from dsvr: {e}")
            klines = asrt.klines(codes, kltype=101, length=32)
            chgklt = {}
            for c in klines:
                chgklt[c] = klPad.cache(c, klines[c], kltype=101)
        finally:
            await self.notify_change(chgklt)


class KlineJobProcess(JobProcess):
    def __init__(self, task_queue, result_queue, period, klt):
        super().__init__(task_queue, result_queue, period)
        self.klt = klt

    def process_prepare(self):
        srtcfg = Config.stockrt_config()
        if 'default_sources' in srtcfg:
            srtapis = ['mklines', 'dklines']
            for _api in srtapis:
                if _api in srtcfg['default_sources']:
                    _val = srtcfg['default_sources'][_api]
                    asrt.set_default_sources(_api, _val[0], tuple(_val[1]), _val[2])
        asrt.set_array_format(srtcfg.get('array_format', 'df'))

    def process_job(self, codes: List[str]) -> Dict:
        return asrt.klines(codes, kltype=self.klt, length=32)


class Stock_Klinem_Watcher(SubProcess_Watcher_Cycle, Stock_Rt_Watcher):
    def __init__(self, m=1, btime=None, etime=None, brks=None):
        if btime is None:
            btime = '9:31:05'
        if etime is None:
            etime = '14:57:1'
        if brks is None:
            brks = [['11:30:1', '13:00:50']]
        super().__init__(m*60, btime, etime, brks)
        Stock_Rt_Watcher.__init__(self)
        self.klt = m

    def create_subprocess(self):
        return KlineJobProcess(self.task_queue, self.result_queue, self.period, self.klt)

    def feed_process_data(self):
        codes = [c for c in self.codes if self.codes[c] > 0]
        if codes:
            self.task_queue.put(codes)

    async def handle_process_result(self, result):
        if not result:
            logger.warning("Received empty kline data")
            return

        try:
            chgklt = {}
            for c in result:
                chgklt[c] = klPad.cache(c, result[c], kltype=self.klt)
            if chgklt:
                await self.notify_change(chgklt)
        except Exception as e:
            logger.error(f"Error processing kline data: {e}")
            logger.error(traceback.format_exc())
            # TODO: 如果数据处理出错，可以考虑重启进程


class QuoteJobProcess(JobProcess):
    def process_prepare(self):
        srtcfg = Config.stockrt_config()
        if 'default_sources' in srtcfg:
            srtapis = ['quotes', 'quotes5']
            for _api in srtapis:
                if _api in srtcfg['default_sources']:
                    _val = srtcfg['default_sources'][_api]
                    asrt.set_default_sources(_api, _val[0], tuple(_val[1]), _val[2])
        asrt.set_array_format(srtcfg.get('array_format', 'df'))

    def process_job(self, codes: List[str]) -> Dict:
        return asrt.quotes(codes)


class Stock_Quote_Watcher(SubProcess_Watcher_Cycle, Stock_Rt_Watcher):
    def __init__(self):
        super().__init__(5, '9:30:1', '14:57', [('11:30', '13:00')])
        Stock_Rt_Watcher.__init__(self)

    def create_subprocess(self):
        return QuoteJobProcess(self.task_queue, self.result_queue, self.period)

    def feed_process_data(self):
        codes = [c for c in self.codes if self.codes[c] > 0]
        if codes:
            self.task_queue.put(codes)

    async def handle_process_result(self, result):
        if not result:
            logger.warning("Received empty quote data")
            return

        try:
            codes = []
            for c in result:
                if result[c] is None:
                    continue
                klPad.cache(c, quotes=result[c])
                codes.append(c)
            await self.notify_change(codes)
        except Exception as e:
            logger.error(f"Error processing quote data: {e}")
            logger.error(traceback.format_exc())
            # TODO: 如果数据处理出错，可以考虑重启进程


class WatcherFactory:
    @staticmethod
    @lru_cache(maxsize=None)
    def get_watcher(name) -> Watcher_Once:
        if name == 'aucsnaps':
            return AuctionSnapshot_Watcher()
        if name == 'stkchanges':
            return StkChanges_Watcher()
        if name == 'bkchanges':
            return BKChanges_Watcher()
        if name == 'stkzdf':
            return StkZdf_Watcher()
        if name == 'end_fundflow':
            return EndFundFlow_Watcher()
        if name == 'kline1':
            return Stock_Klinem_Watcher(1)
        if name == 'kline15':
            return Stock_Klinem_Watcher(15, '9:44:55', '14:57:1', [('11:30:1', '13:14:50')])
        if name == 'klineday':
            return Stock_KlineDay_Watcher()
        if name == 'quotes':
            return Stock_Quote_Watcher()
        raise ValueError(f"Unknown watcher: {name}")
