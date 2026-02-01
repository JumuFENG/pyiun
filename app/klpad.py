import time
import datetime
import json
from functools import lru_cache
import stockrt as srt
from stockrt.sources.rtbase import requestbase
from app.guang import guang
from app.lofig import logger


class klPad:
    __stocks = {}
    __factors = [2, 4, 8]

    @classmethod
    def dump(self):
        # return self.__stocks
        return {c: v for c,v in self.__stocks.items() if 'klines' in v and 15 in v['klines']}

    @classmethod
    def cache(self, code, klines=[], quotes={}, kltype=1):
        if code not in self.__stocks:
            self.__stocks[code] = {
                'klines': {},
                'quotes': {}
            }
        if quotes:
            self.__stocks[code]['quotes'].update(quotes)
            if 'bid5' in quotes:
                self.__stocks[code]['quotes']['q5time'] = time.time()

        if len(klines) == 0:
            return []
        mcount = self.merge_klines(code, kltype, klines)
        if mcount > 0:
            efacs = self.expand_kltypes(code, kltype)
            fac = [1] + efacs
            return [kltype * f for f in fac]
        return []

    @classmethod
    def merge_klines(cls, code: str, kltype: int, klines) -> int:
        """
        将K线数据合并到存储中（list-of-dicts 版本）
        """
        # 规范化输入为 list
        if not isinstance(klines, list):
            klines = list(klines)

        if kltype <= 60 and len(klines) >= 2:
            last_kl = klines[-1]
            try:
                last_time_dt = datetime.datetime.fromisoformat(last_kl['time'])
            except Exception:
                last_time_dt = datetime.datetime.strptime(last_kl['time'], '%Y-%m-%d %H:%M:%S')
            time_diff = (last_time_dt - datetime.datetime.now()).total_seconds()
            if time_diff > kltype * 60 * 0.8:
                prev = klines[-2]
                prev['close'] = last_kl['close']
                prev['high'] = max(prev.get('high', 0), last_kl.get('high', 0))
                prev['low'] = min(prev.get('low', 1e12), last_kl.get('low', 1e12))
                prev['volume'] = prev.get('volume', 0) + last_kl.get('volume', 0)
                if 'amount' in last_kl or 'amount' in prev:
                    prev['amount'] = prev.get('amount', 0) + last_kl.get('amount', 0)
                klines = klines[:-1]

        if kltype == 1:
            # 合并 09:30 到 09:31 的集合竞价
            remove_idx = set()
            for i, item in enumerate(klines):
                if item['time'].endswith('09:30') and i + 1 < len(klines) and klines[i+1]['time'].endswith('09:31'):
                    klines[i+1]['volume'] = klines[i+1].get('volume', 0) + item.get('volume', 0)
                    if 'amount' in item:
                        klines[i+1]['amount'] = klines[i+1].get('amount', 0) + item.get('amount', 0)
                    remove_idx.add(i)
            if remove_idx:
                klines = [it for idx, it in enumerate(klines) if idx not in remove_idx]

        stored_klines = cls.__stocks[code]['klines'].get(kltype, [])
        if len(stored_klines) == 0:
            # 初始化逻辑
            if kltype == 1:
                start_idx = next((i for i, r in enumerate(klines) if r['time'].endswith(('09:31', '13:01'))), 0)
                klines = klines[start_idx:]
            elif kltype == 15:
                start_idx = next((i for i, r in enumerate(klines) if r['time'].endswith('09:45')), 0)
                klines = klines[start_idx:]

            cls.__stocks[code]['klines'][kltype] = klines
            return len(klines)

        # 增量更新逻辑
        if len(klines) > 0 and len(stored_klines) > 0 and klines[0]['time'] <= stored_klines[-1]['time']:
            stored_klines = stored_klines[:-1]
        last_time = stored_klines[-1]['time'] if len(stored_klines) > 0 else ''
        new_data = [r for r in klines if r['time'] > last_time]

        if len(new_data) > 0:
            cls.__stocks[code]['klines'][kltype] = (stored_klines + new_data) if len(stored_klines) > 0 else new_data
        return len(new_data)

    @classmethod
    def expand_kltypes(cls, code: str, base_kltype: int) -> None:
        base_klines = cls.__stocks[code]['klines'][base_kltype]

        efacs = []
        for fac in cls.__factors:
            ex_kltype = base_kltype * fac

            if ex_kltype not in cls.__stocks[code]['klines']:
                cls.__stocks[code]['klines'][ex_kltype] = []

            ex_klines = cls.__stocks[code]['klines'][ex_kltype]
            last_time = ex_klines[-1]['time'] if len(ex_klines) > 0 else ''

            new_data = [r for r in base_klines if r['time'] > last_time]

            if len(new_data) == 0:
                continue
            new_last_time = new_data[-1]['time'].split()[-1]
            expand_unfinished = base_kltype <= 15 and new_last_time >= '14:56'
            if len(new_data) >= fac or expand_unfinished:
                if len(ex_klines) > 0:
                    ex_klines = ex_klines[:-1]
                    last_time = ex_klines[-1]['time'] if len(ex_klines) > 0 else ''
                    new_data = [r for r in base_klines if r['time'] > last_time]

                expanded_klines = []
                for i in range(0, len(new_data), fac):
                    group = new_data[i:i+fac]
                    if len(group) == fac or expand_unfinished:
                        new_kline = {
                            'time': group[-1]['time'],
                            'open': group[0]['open'],
                            'close': group[-1]['close'],
                            'high': max([g.get('high', 0) for g in group]),
                            'low': min([g.get('low', 1e12) for g in group]),
                            'volume': sum([g.get('volume', 0) for g in group])
                        }
                        if 'amount' in group[0]:
                            new_kline['amount'] = sum([g.get('amount', 0) for g in group])
                        expanded_klines.append(new_kline)

                if expanded_klines:
                    cls.__stocks[code]['klines'][ex_kltype] = (ex_klines + expanded_klines) if len(ex_klines) > 0 else expanded_klines
                    efacs.append(fac)
        return efacs

    @classmethod
    def calc_indicators(self, code, kltype):
        if code not in self.__stocks:
            return
        exkltypes = [kltype]
        if kltype == 1 or kltype == 15:
            exkltypes += [kltype * fa for fa in self.__factors]
        for ex_kltype in exkltypes:
            if ex_kltype not in self.__stocks[code]['klines']:
                continue
            self.calc_ma(code, ex_kltype, 18)
            self.calc_bss(code, ex_kltype, 18)

    @classmethod
    def calc_ma(cls, code: str, kltype: int, n: int) -> None:
        klines = cls.__stocks[code]['klines'][kltype]
        if len(klines) == 0:
            return
        col_name = f'ma{n}'

        closes = [r.get('close', 0) for r in klines]

        # 初始化或重新计算
        if col_name not in klines[0]:
            for i in range(len(klines)):
                window = closes[max(0, i - n + 1): i + 1]
                klines[i][col_name] = sum(window) / len(window) if window else 0
            return

        # 找到最后一个有效ma
        last_valid = -1
        for i in range(len(klines)-1, -1, -1):
            if klines[i].get(col_name) is not None:
                last_valid = i
                break
        start_idx = 0 if last_valid == -1 else last_valid
        start_idx = max(0, start_idx)
        if start_idx <= n:
            for i in range(len(klines)):
                window = closes[max(0, i - n + 1): i + 1]
                klines[i][col_name] = sum(window) / len(window) if window else 0
            return

        for i in range(start_idx, len(klines)):
            window = closes[max(0, i - n + 1): i + 1]
            klines[i][col_name] = sum(window) / len(window) if window else 0

    @classmethod
    def calc_bss(cls, code: str, kltype: int, n: int) -> None:
        klines = cls.__stocks[code]['klines'][kltype]
        if len(klines) < 2:
            return

        col_name = f'bss{n}'
        ma_col = f'ma{n}'

        # 初始化 bss 列
        if col_name not in klines[0]:
            for r in klines:
                r[col_name] = None

        # 确定起始位置
        last_valid = -1
        for i in range(len(klines)-1, -1, -1):
            if klines[i].get(col_name) is not None:
                last_valid = i
                break
        start_idx = 2 if last_valid == -1 else last_valid

        if start_idx >= len(klines):
            return

        updates = {}
        prev_bss = 'u' if start_idx == 2 else klines[start_idx-1].get(col_name, 'u')

        for i in range(start_idx, len(klines)):
            cur = klines[i]
            prev = klines[i-1]
            ma = cur.get(ma_col, 0)
            above_i = (cur.get('low', 0) > ma) or ((min(cur.get('open', 0), cur.get('close', 0)) > ma) and ((cur.get('high',0) - cur.get('low',0)) * 0.8 <= abs(cur.get('open',0) - cur.get('close',0))))
            above_prev = (prev.get('low',0) > prev.get(ma_col, 0)) or ((min(prev.get('open',0), prev.get('close',0)) > prev.get(ma_col,0)) and ((prev.get('high',0) - prev.get('low',0)) * 0.8 <= abs(prev.get('open',0) - prev.get('close',0))))

            below_i = (cur.get('high', 0) < ma) or ((max(cur.get('open', 0), cur.get('close', 0)) < ma) and ((cur.get('high',0) - cur.get('low',0)) * 0.8 <= abs(cur.get('open',0) - cur.get('close',0))))
            below_prev = (prev.get('high',0) < prev.get(ma_col,0)) or ((max(prev.get('open',0), prev.get('close',0)) < prev.get(ma_col,0)) and ((prev.get('high',0) - prev.get('low',0)) * 0.8 <= abs(prev.get('open',0) - prev.get('close',0))))

            if above_i and above_prev:
                new_bss = 'b' if prev_bss in ('u', 'w') else 'h'
            elif below_i and below_prev:
                new_bss = 's' if prev_bss in ('u', 'h') else 'w'
            else:
                new_bss = 'h' if prev_bss in ('b', 'h') else 'w' if prev_bss in ('s', 'w') else 'u'

            updates[i] = new_bss
            prev_bss = new_bss

        for idx, val in updates.items():
            klines[idx][col_name] = val

    @classmethod
    def get_klines(self, code, kltype=1):
        if code not in self.__stocks or kltype not in self.__stocks[code]['klines']:
            return []
        return self.__stocks[code]['klines'][kltype]

    @classmethod
    @lru_cache(maxsize=1)
    def dsvr_source(self):
        return DsvrKSource()

    @classmethod
    def load_dsvr_klines(self, stocks, kltype=101, length=1000, fq=1):
        if not stocks:
            return {}
        klines = self.dsvr_source().klines(stocks, kltype, length, fq)
        chgklt = {}
        for c, k in klines.items():
            chgklt[c] = self.cache(c, k, kltype=kltype)
        return chgklt

    @classmethod
    def load_dsvr_quotes(self, stocks):
        if not stocks:
            return []
        quotes = self.dsvr_source().quotes(stocks)
        codes = []
        for c, q in quotes.items():
            self.cache(c, quotes=q)
            codes.append(c)
        return codes

    @classmethod
    def resize_cached_klines(self, code, n):
        if code not in self.__stocks:
            return
        for kltype in list(self.__stocks[code]['klines'].keys()):
            arr = self.__stocks[code]['klines'][kltype]
            if isinstance(arr, list):
                self.__stocks[code]['klines'][kltype] = arr[-n:]
            else:
                # backward compatibility
                self.__stocks[code]['klines'][kltype] = list(arr)[-n:]

    @classmethod
    def get_quotes(self, code):
        if code not in self.__stocks:
            return {}
        return self.__stocks[code]['quotes']

    @classmethod
    def get_quotes5(self, code):
        if code not in self.__stocks or 'q5time' not in self.__stocks[code]['quotes'] or time.time() - self.__stocks[code]['quotes']['q5time'] > 3:
            q5 = srt.quotes5(code)
            if not q5:
                return {}
            self.cache(code, quotes=q5[code])
        return self.__stocks[code]['quotes']

    @classmethod
    def get_lclose_from_klines(self, code):
        if code not in self.__stocks:
            return 0
        today = guang.today_date('-')
        for klines in self.__stocks[code]['klines'].values():
            if len(klines) > 0:
                for i, r in enumerate(klines):
                    if r['time'].startswith(today):
                        if i > 0:
                            return klines[i-1]['close']
                        break
        return 0

    @classmethod
    def get_zt_price(self, code):
        if code not in self.__stocks:
            return 0
        quotes = self.__stocks[code]['quotes']
        if 'top_price' not in quotes:
            lclose = quotes['lclose'] if 'lclose' in quotes else self.get_lclose_from_klines(code)
            if lclose == 0:
                return 0
            return guang.zt_priceby(lclose, zdf=guang.zdf_from_code(code))
        return quotes['top_price']

    @classmethod
    def get_dt_price(self, code):
        if code not in self.__stocks:
            return 0
        quotes = self.__stocks[code]['quotes']
        if 'bottom_price' not in quotes:
            lclose = quotes['lclose'] if 'lclose' in quotes else self.get_lclose_from_klines(code)
            if lclose == 0:
                return 0
            return guang.dt_priceby(lclose, zdf=guang.zdf_from_code(code))
        return quotes['bottom_price']

    @staticmethod
    def continuously_increase_days(code, kltype):
        klines = klPad.get_klines(code, kltype)
        if len(klines) == 0:
            return 0

        n = 0
        closes = [r.get('close', 0) for r in klines]

        for i in range(len(closes)-1, 0, -1):
            if closes[i] < closes[i-1]:
                break
            if closes[i] == closes[i-1]:
                continue
            n += 1

        return n

    @staticmethod
    def continuously_dt_days(code, yz=False):
        klines = klPad.get_klines(code, 101)
        if len(klines) == 0:
            return 0

        n = 0
        highs = [r.get('high', 0) for r in klines]
        lows = [r.get('low', 0) for r in klines]
        closes = [r.get('close', 0) for r in klines]
        for i in range(len(closes) - 1, 0, -1):
            if yz and highs[i] - lows[i] > 0:
                break
            if closes[i] <= guang.dt_priceby(closes[i-1], zdf=guang.zdf_from_code(code)):
                n += 1
        return n

    @staticmethod
    def get_last_trough(code, kltype):
        klines = klPad.get_klines(code, kltype)
        if len(klines) == 0:
            return 0

        lows = [r.get('low', 0) for r in klines]
        down_num = 0
        up_num = 0
        tprice = lows[-1]
        for i in range(len(lows) - 1, 0, -1):
            if down_num < 2:
                if lows[i] < lows[i-1]:
                    continue
                if lows[i] > lows[i-1]:
                    down_num += 1
                    tprice = lows[i-1]
            else:
                if lows[i] > lows[i-1]:
                    if up_num >= 2:
                        break
                    if tprice > lows[i-1]:
                        down_num += 1
                        tprice = lows[i-1]
                    up_num = 0
                    continue
                if lows[i] < lows[i-1]:
                    up_num += 1
        if up_num >= 2 and down_num > 2:
            return tprice
        return 0


class DsvrKSource(requestbase):
    dserver = 'http://localhost/5000/'
    def __init__(self):
        super().__init__()

    @property
    def qtapi(self):
        return self.dserver + 'stock/quotes?code=%s'

    @property
    def qt5api(self):
        pass

    @property
    def tlineapi(self):
        return self.dserver + 'stock/tlines?code=%s'

    @property
    def mklineapi(self):
        return self.dserver + 'stock/klines?code=%s&kltype=%s&fqt=%s&length=%s'

    @property
    def dklineapi(self):
        return self.mklineapi

    @property
    def fklineapi(self):
        return self.mklineapi

    def get_quote_url(self, stocks):
        return self.qtapi % ','.join(stocks), self._get_headers()

    def format_quote_response(self, rep_data):
        result = {}
        for codes, d in rep_data:
            data = json.loads(d)
            if not data:
                continue
            for stock in data:
                code = stock if stock in codes else stock[-6:] if stock[-6:] in codes else stock
                result[code] = data[stock]
        return result

    def get_tline_url(self, stock):
        return self.tlineapi % ','.join(stock), self._get_headers()

    def format_tline_response(self, rep_data):
        result = {}
        for codes, d in rep_data:
            data = json.loads(d)
            if not data:
                continue
            for stock in data:
                code = stock if stock in codes else stock[-6:] if stock[-6:] in codes else stock
                result[code] = data[stock]
        return result

    def tlines(self, stocks):
        stocks = self._stock_groups(stocks)
        return self._fetch_concurrently(stocks, self.get_tline_url, self.format_tline_response)

    def get_mkline_url(self, stock, kltype='1', length=320, fq=1):
        kltype = self.to_int_kltype(kltype)
        return self.mklineapi % (','.join(stock), kltype, fq, length), self._get_headers()

    def get_dkline_url(self, stock, kltype='101', length=320, fq=1):
        return self.get_mkline_url(stock, kltype, length, fq)

    def get_fkline_url(self, stock, kltype='101', fq=0):
        kltype = self.to_int_kltype(kltype)
        return self.fklineapi % (stock, kltype, fq), self._get_headers()

    def format_kline_response(self, rep_data, **kwargs):
        result = {}
        for codes, d in rep_data:
            data = json.loads(d)
            if not data:
                continue
            for stock in data:
                code = stock if stock in codes else stock[-6:] if stock[-6:] in codes else stock
                cols = ['time', 'open', 'close', 'high', 'low', 'volume', 'amount', 'change', 'change_px', 'amplitude', 'turnover']
                if len(data[stock][0]) < 11:
                    cols = cols[:len(data[stock][0])]
                karr = []
                for item in data[stock]:
                    karr.append(item[:len(cols)])
                result[code] = self.format_array_list(karr, cols)
        return result

    def mklines(self, stocks, kltype, length=320, fq=1, withqt=False):
        if not self.mklineapi:
            return
        kltype = self.to_int_kltype(kltype)
        stocks = self._stock_groups(stocks)
        return self._fetch_concurrently(stocks, self.get_mkline_url, self.format_kline_response, url_kwargs={'kltype': kltype, 'length': length, 'fq': fq}, fmt_kwargs={'is_minute': True, 'withqt': False})

    def dklines(self, stocks, kltype=101, length=320, fq=1, withqt=False):
        return self.mklines(stocks, kltype, length, fq, withqt)

    def fklines(self, stocks, kltype=101, fq=0):
        return self.mklines(stocks, kltype, length=0, fq=fq, withqt=False)

    def klines(self, stocks, kltype = 1, length=320, fq=1):
        return self.mklines(stocks, kltype, length, fq, withqt=False)