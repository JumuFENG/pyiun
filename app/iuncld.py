import json
import emxg
import stockrt as asrt
from functools import lru_cache
from threading import Lock
from datetime import datetime, timedelta
from stockrt.sources.eastmoney import Em
from app.trade_interface import TradeInterface
from app.lofig import logger
from app.guang import guang
from app.klpad import klPad



class iunCloud:
    dserver = None

    __save_db = True
    @classmethod
    def disable_save_db(cls):
        cls.__save_db = False

    @classmethod
    def save_db_enabled(cls):
        return cls.__save_db

    @staticmethod
    def iun_str_conf(ikey):
        return TradeInterface.iun_str()[ikey]

    @staticmethod
    def is_rzrq(code):
        if TradeInterface.tserver is None:
            return False
        return TradeInterface.is_rzrq(code)

    @staticmethod
    def get_account_latest_stocks(account):
        return TradeInterface.get_account_latest_stocks(account)

    @staticmethod
    def get_hold_account(code, account):
        if account == '':
            return 'collat' if iunCloud.is_rzrq(code) else 'normal'
        if account == 'credit':
            return 'collat'
        return account

    __bk_ignored = None
    @classmethod
    def is_bk_ignored(self, bk):
        if self.__bk_ignored is None:
            url = guang.join_url(iunCloud.dserver, 'stock')
            params = {
                'act': 'bk_ignored',
            }
            self.__bk_ignored = json.loads(guang.get_request(url, params=params))
        return bk in self.__bk_ignored

    @staticmethod
    @lru_cache(maxsize=1)
    def black_list():
        # ST股 B股
        return iunCloud.get_bk_stocks('BK0511') + iunCloud.get_bk_stocks('BK0636')

    @classmethod
    def is_stock_blacked(self, code):
        return code[-6:] in self.black_list()

    __dividen = None
    @classmethod
    def to_be_divided(self, code):
        if self.__dividen is None:
            url = guang.join_url(iunCloud.dserver, 'stock')
            params = {
                'act': 'planeddividen',
                'date': guang.today_date('-')
            }
            dividedetails = json.loads(guang.get_request(url, params=params))
            d35 = (datetime.now() + (timedelta(days=2) if datetime.now().day < 3 else timedelta(days=4))).strftime('%Y-%m-%d')
            self.__dividen = [d[1][-6:] for d in dividedetails if d[3] <= d35]
        return code[-6:] in self.__dividen

    __stock_bks = {}
    @classmethod
    def get_stock_bks(self, code):
        code = code[-6:]
        if code not in self.__stock_bks:
            url = guang.join_url(iunCloud.dserver, 'stock')
            params = {
                'act': 'stockbks',
                'stocks': code
            }
            bks = json.loads(guang.get_request(url, params=params))
            for c, b in bks.items():
                self.__stock_bks[c[-6:]] = [_b[0] for _b in b]
        return self.__stock_bks[code]

    @classmethod
    @lru_cache(maxsize=100)
    def get_bk_stocks(self, bk):
        url = guang.join_url(iunCloud.dserver, 'stock')
        params = {
            'act': 'bkstocks',
            'bks': bk
        }
        stks = json.loads(guang.get_request(url, params=params))
        return stks[bk] if bk in stks else []

    @staticmethod
    @lru_cache(maxsize=1)
    def zt_recently():
        url = guang.join_url(iunCloud.dserver, 'stock')
        params = {
            'act': 'ztstocks',
            'days': 3
        }
        return [c[-6:] for c in json.loads(guang.get_request(url, params=params))]

    @classmethod
    def recent_zt(self, code):
        return code[-6:] in self.zt_recently()

    @staticmethod
    @lru_cache(maxsize=1)
    def topbks5():
        url = guang.join_url(iunCloud.dserver, 'stock')
        params = {
            'act': 'hotbks',
            'days': 5
        }
        rsp = guang.get_request(url, params=params)
        return json.loads(rsp)

    @classmethod
    def is_topbk5(self, bk):
        return bk in self.topbks5()

    @staticmethod
    @lru_cache(maxsize=10)
    def get_hotstocks(days=5):
        url = guang.join_url(iunCloud.dserver, 'stock')
        params = {
            'act': 'hotstocks',
            'days': days
        }
        return json.loads(guang.get_request(url, params=params))

    @staticmethod
    def get_stock_fflow(code, date=None, date1=None):
        """
        Get the stock's main fund flow data from eastmoney.

        Parameters:
            code (str): The stock code, e.g. '600777'.

            date (str): The start date of the data, e.g. '2025-04-09'.

            date1 (str): The end date of the data, e.g. '2025-04-09'.

        Returns:
        list: A list of lists, each contains the date and the main fund flow data of the stock.
        [日期, 主力, 小单, 中单, 大单, 超大单 (净流入/占比)]
        [['2025-04-09', '-18626339.0', '7829801.0', '10796540.0', '15861618.0', '-34487957.0', '-5.70', '2.39', '3.30', '4.85', '-10.55', '2.66', '0.76', '0.00', '0.00']]
        Example:
        >>> iunCloud.get_stock_fflow('600777', '2025-04-09', '2025-04-09')
        """
        url = guang.join_url(iunCloud.dserver, 'stock_fflow')
        params = {
            'code': code,
            'date': date,
        }

        fflow = json.loads(guang.get_request(url, params=params))
        if isinstance(fflow[0][0], str):
            return [f for f in fflow if date is None or f[0] >= date and (date1 is None or f[0] <= date1)]
        values = [f[1:] for f in fflow if date is None or f[1] >= date and (date1 is None or f[1] <= date1)]
        return values

    @staticmethod
    def get_emxg_stock_zdfrank(minzdf=None):
        if minzdf is None:
            return []

        pdata = emxg.search(keyword=f'涨跌幅>{minzdf}%' if minzdf > 0 else f'涨跌幅<{minzdf}%')
        pdata = pdata.rename(columns = {
            '代码': 'code', '名称': 'name', '股票简称': 'name', '最新价': 'close', '涨跌额': 'change_px', '涨跌幅': 'change',
            '涨跌幅:前复权': 'change', '成交量(股)': 'volume', '成交量': 'volume', '开盘价:前复权': 'open', '开盘价': 'open',
            '最高价:前复权': 'high', '最高价(日线不复权)': 'high', '最低价:前复权': 'low', '最低价(日线不复权)': 'low', '成交额': 'amount'
        })
        pdata = pdata.assign(code=lambda x: x['code'].apply(asrt.get_fullcode) if hasattr(x['code'], 'apply') else asrt.get_fullcode(x['code']))
        if 'lclose' not in pdata.columns:
            if 'change_px' not in pdata.columns:
                pdata = pdata.assign(lclose=lambda x: x['close'] / (1 + x['change']))
                pdata = pdata.assign(change_px=lambda x: x['close'] - x['lclose'])
            else:
                pdata = pdata.assign(lclose=lambda x: x['close'] - x['change_px'])

        if 'amount' not in pdata.columns:
            pdata = pdata.assign(amount=lambda x: x['close'] * x['volume'])
        if 'open' not in pdata.columns:
            pdata = pdata.assign(open=lambda x: x['lclose'])
        result = pdata[['code', 'name', 'close', 'high', 'low', 'open', 'change', 'volume', 'amount', 'change_px', 'lclose']].to_dict('records')
        return result

    @staticmethod
    def get_stocks_zdfrank(minzdf=None):
        if minzdf is None:
            stocks = asrt.stock_list()
            if stocks is None or 'all' not in stocks:
                return []
            return stocks['all']

        try:
            result = iunCloud.get_emxg_stock_zdfrank(minzdf=minzdf)
            if not result:
                clist = Em.qt_clist(
                    fs='m:0+t:6+f:!2,m:0+t:80+f:!2,m:1+t:2+f:!2,m:1+t:23+f:!2,m:0+t:81+s:262144+f:!2',
                    fields='f1,f2,f3,f4,f5,f6,f15,f16,f17,f18,f12,f13,f14',
                    fid='f3', po=1 if minzdf > 0 else 0,
                    qtcb=lambda data: any(abs(d['f3']) < abs(minzdf) for d in data)
                )
                if not clist:
                    raise Exception('No data from Em.qt_clist')
                result = [{
                    'code': asrt.get_fullcode(s['f12']),
                    'name': s['f14'],
                    'close': float(s['f2']),
                    'high': float(s['f15']) if s['f15'] != '-' else 0,
                    'low': float(s['f16']) if s['f16'] != '-' else 0,
                    'open': float(s['f17']) if s['f17'] != '-' else 0,
                    'lclose': float(s['f18']),
                    'change_px': float(s['f4']),
                    'change': float(s['f3']) / 100,
                    'volume': (int(s['f5']) if s['f5'] != '-' else 0) * 100,
                    'amount': float(s['f6']) if s['f6'] != '-' else 0
                } for s in clist if s['f2'] != '-' and s['f18'] != '-']
        except Exception as e:
            logger.warning(f'get_stocks_zdfrank error: {e}')

            clist = iunCloud.get_stocks_zdfrank()
            minzdf /= 100
            if minzdf < 0:
                clist = [s for s in clist if s['change'] <= minzdf]
                clist = list(reversed(clist))
            elif minzdf > 0:
                clist = [s for s in clist if s['change'] >= minzdf]

            result = [{
                'code': asrt.get_fullcode(s['code']),
                'name': s['name'],
                'close': float(s['close']),
                'high': float(s.get('high', 0)),
                'low': float(s.get('low', 0)),
                'open': float(s.get('open', 0)),
                'lclose': float(s.get('lclose', 0)),
                'change_px': float(s.get('change_px', 0)),
                'change': float(s.get('change', 0)),
                'volume': int(s.get('volume', 0)),
                'amount': float(s.get('amount', 0))
            } for s in clist]
        return result

    @staticmethod
    def get_zdfb():
        try:
            fburl = 'https://push2ex.eastmoney.com/getTopicZDFenBu?ut=7eea3edcaed734bea9cbfc24409ed989&dpt=wz.ztzt'
            zdfb = json.loads(guang.get_request(fburl, headers=guang.em_headers(Host='push2ex.eastmoney.com')))['data']['fenbu']
            fbdic = {}
            [fbdic.update(d) for d in zdfb]
            up = sum([fbdic[k] for k in fbdic if int(k) > 0])
            down = sum([fbdic[k] for k in fbdic if int(k) < 0])
            fbdic.update({'up': up, 'down': down})
            return fbdic
        except:
            logger.error('get_zdfb error')

    __ranklock = Lock()
    @classmethod
    @lru_cache(maxsize=1)
    def get_open_hotranks(cls):
        with cls.__ranklock:
            emrk_url = ('http://datacenter-web.eastmoney.com/wstock/selection/api/data/get?'
                'type=RPTA_PCNEW_STOCKSELECT&sty=POPULARITY_RANK,NEWFANS_RATIO&filter='
                '(POPULARITY_RANK>0)(POPULARITY_RANK<=100)(NEWFANS_RATIO>=0.00)(NEWFANS_RATIO<=100.0)'
                '&p=1&ps=100&st=POPULARITY_RANK&sr=1&source=SELECT_SECURITIES&client=WEB')
            jdata = json.loads(guang.get_request(emrk_url))
            if not jdata or jdata['code'] != 0 or 'result' not in jdata or 'data' not in jdata['result']:
                rkurl = guang.join_url(iunCloud.dserver, 'stock?act=hotrankrt&rank=40')
                jdata = [ {'SECURITY_CODE': x[0], 'POPULARITY_RANK': x[1], 'NEWFANS_RATIO': x[2]} for x in json.loads(guang.get_request(rkurl))]
            else:
                jdata = jdata['result']['data']

            rkdict = {}
            for rk in jdata:
                code = rk['SECURITY_CODE']
                rkdict[code] = {'rank': rk['POPULARITY_RANK'], 'newfans': rk['NEWFANS_RATIO']}
            return rkdict

    @classmethod
    @lru_cache(maxsize=1)
    def get_suspend_stocks(cls):
        url = ("https://datacenter-web.eastmoney.com/api/data/v1/get?sortColumns=SUSPEND_START_DATE&sortTypes=-1&pageSize=500&pageNumber=1"
        f'''&reportName=RPT_CUSTOM_SUSPEND_DATA_INTERFACE&columns=ALL&source=WEB&client=WEB&filter=(MARKET="全部")(DATETIME='{guang.today_date('-')}')''')
        try:
            sus = json.loads(guang.get_request(url))
            return tuple(s['SECURITY_CODE'] for s in sus['result']['data'])
        except Exception as e:
            logger.error('get_suspend_stocks error: %s', e)
            cls.get_suspend_stocks.cache_clear()
            return tuple()

    @classmethod
    @lru_cache(maxsize=1)
    def get_dailyzdt(cls):
        '''
        获取最近10天的涨停数据
        Returns:
        [date, ztcount, zt0cnt, dtcount]
        '''
        url = guang.join_url(iunCloud.dserver, 'stock?act=zdtemot&days=10')
        return json.loads(guang.get_request(url))

    @classmethod
    @lru_cache(maxsize=1)
    def get_dailyztsteps_gt3(cls):
        '''
        最近3天涨停连板次数大于等于4的个数
        '''
        url = guang.join_url(iunCloud.dserver, 'stock?act=ztstepshist&days=3&steps=4')
        return json.loads(guang.get_request(url))

    @classmethod
    @lru_cache(maxsize=1)
    def get_financial_4season_losing(cls):
        try:
            pdata = emxg.search(keyword='连续4个季度亏损大于1000万')
            pdata = pdata.rename(columns = {'代码': 'code'})
            logger.info('连续4个季度亏损大于1000万: %d', len(pdata))
            return tuple(pdata['code'])
        except Exception as e:
            logger.info('emxg.search error: %s', e)
            url = guang.join_url(iunCloud.dserver, 'stock?act=f4lost')
            return tuple([c[-6:] for c in json.loads(guang.get_request(url))])

    @classmethod
    @lru_cache(maxsize=1)
    def get_financial_cheating(cls):
        try:
            pdata = emxg.search(keyword='财务造假')
            pdata = pdata.rename(columns = {'代码': 'code'})
            logger.info('财务造假: %d', len(pdata))
            return tuple(pdata['code'])
        except Exception as e:
            logger.info('emxg.search error: %s', e)
            return tuple()

    @classmethod
    def financial_block(self, code):
        if code in self.get_financial_cheating():
            return True

        if code in self.get_financial_4season_losing():
            return True

        quote = klPad.get_quotes(code)
        if 'name' not in quote:
            return False
        if quote['name'].startswith('退市') or 'ST' in quote['name'] or quote['name'].endswith('退'):
            return True
        # if 'PE' in quote and quote['PE'] < 0:
        #     return True
        # if 'PB' in quote and quote['PB'] < 1:
        #     return True
        # if 'TTM_PE' in quote and quote['TTM_PE'] < 0:
        #     return True
        return False

