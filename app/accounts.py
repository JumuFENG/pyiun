import json
from stockrt import get_fullcode, quotes
from app.lofig import logger
from app.guang import guang
from app.iuncld import iunCloud
from app.klpad import klPad
from app.stock_strategy import StockStrategyFactory as sfac


class Account(object):
    recycle_strs = ['StrategyGE', 'StrategyBSBE']
    def __init__(self, acc):
        self.keyword = acc
        self.trading_remarks = {}

    def load_watchings(self) -> None:
        surl = f"{accld.dserver}stock?act=watchings&acc={self.keyword}"
        stocks = guang.get_request_json(surl, headers=accld.headers)
        for c, v in stocks.items():
            if c in iunCloud.get_suspend_stocks():
                logger.info('%s is suspended', c)
                continue
            code = c[-6:]
            if not v['strategies']['strategies']:
                logger.error('%s %s has no strategy', self.keyword, c)
                continue

            sfac.cache_stock_strategy(self.keyword, code, v)

            for sobj in v['strategies']['strategies'].values():
                if not sobj['enabled']:
                    continue
                s = sfac.get_strategy(sobj['key'])
                if s:
                    s.add_stock(self.keyword, code)
        logger.info('%s Loaded stocks: %d', self.keyword, len(stocks))

    def verify_strategies(self):
        today = guang.today_date('-')
        stocks = iunCloud.get_account_latest_stocks(self.keyword)
        holding = [s['code'] for s in stocks if s['holdCount'] > 0 and 'strategies' in s]
        if holding:
            logger.info('%s trading remarks %s', self.keyword, self.trading_remarks)
            hquotes = quotes(holding)
            for c in hquotes:
                klPad.cache(c, quotes=hquotes[c])

        for stock in stocks:
            if 'strategies' not in stock:
                logger.info('%s %s has no strategy %s', self.keyword, stock['code'], stock)
                continue
            if stock['holdCount'] == 0 and ('strategies' not in stock['strategies'] or not stock['strategies']['strategies']):
                logger.info('%s %s holdCount = 0 %s', self.keyword, stock['code'], stock)
                continue
            code = stock['code']
            buydetails = stock['strategies'].get('buydetail', [])
            buydetails_full = stock['strategies'].get('buydetail_full', [])
            if self.keyword in ('normal', 'collat'):
                logger.info('%s %s %s %s', self.keyword, code, buydetails, buydetails_full)
            traded = False
            for rec in buydetails_full:
                if rec['date'].split(' ')[0] == today and rec['count'] > 0:
                    traded = True
                    break

            if 'strategies' in stock['strategies']:
                dkeys = []
                for k, sobj in stock['strategies']['strategies'].items():
                    if sobj['key'] == 'StrategyBuyZTBoard':
                        dkeys.append(k)
                        continue

                if dkeys:
                    for k in dkeys:
                        del stock['strategies']['strategies'][k]

            if not traded:
                self.verify_not_traded(stock)
                continue

            count = stock['holdCount']
            if count == 0:
                self.verify_recycle_strategies(stock)
                continue
            if self.verify_track_buysell(stock):
                continue
            if self.verify_zt_top_reached(stock):
                continue

            if code in self.trading_remarks:
                if self.trading_remarks[code] == 'istrategy_zt1bk':
                    self.verify_zt1bk_strategies(stock)
                if self.trading_remarks[code] == 'istrategy_hsrzt0':
                    self.verify_hrszt0_strategies(stock)
                if self.trading_remarks[code] == 'istrategy_hotstks_open':
                    self.verify_hotstks_open_strategies(stock)
                if self.trading_remarks[code] == 'istrategy_zt1wb':
                    self.verify_zt1wb_strategies(stock)
                continue
            if 'strategies' in stock['strategies']:
                dkeys = []
                for k, sobj in stock['strategies']['strategies'].items():
                    if sobj['key'] == 'StrategyBSBE':
                        smeta = sfac.get_strategy_meta(self.keyword, code, sobj['key'])
                        if not smeta:
                            continue
                        logger.info('set guardPrice for %s with meta %s', code, smeta)
                        sobj['guardPrice'] = smeta['guardPrice']
                        continue
                    if sobj['key'] == 'StrategyGE':
                        smeta = sfac.get_strategy_meta(self.keyword, code, sobj['key'])
                        if not smeta:
                            continue
                        if 'guardPrice' in smeta:
                            if 'lost' in smeta:
                                sobj['guardPrice'] = round((sum([ b['price']*b['count'] for b in buydetails]) + smeta['lost']) / sum([b['count'] for b in buydetails]), 2)
                                if 'lost' in sobj:
                                    del sobj['lost']
                                logger.info('set guardPrice for %s with meta %s', code, smeta)
                            else:
                                sobj['guardPrice'] = smeta['guardPrice']
                        elif 'guardPrice' in sobj:
                            del sobj['guardPrice']
                        continue

            self.save_stock_strategy(code, stock['strategies'])

    def verify_not_traded(self, stock):
        if stock['holdCount'] <= 0:
            return False
        return self.verify_zt_top_reached(stock)

    def create_strategy_keep_recycles(self, code, smeta, origin_strategies):
        strategies = guang.generate_strategy_json({'code': code, 'strategies': smeta}, {'amount': origin_strategies.get('amount', 10000)})
        strategies['buydetail'] = origin_strategies.get('buydetail', [])
        strategies['buydetail_full'] = origin_strategies.get('buydetail_full', [])
        ikey = len(smeta)
        for s in self.recycle_strs:
            sobj = sfac.get_strategy_meta(self.keyword, code, s)
            if sobj:
                strategies['strategies'][ikey] = sobj
                ikey += 1
        return strategies

    def verify_zt_top_reached(self, stock):
        '''
        如果当日涨停或接近目标价，调整卖出策略
        '''

        code = stock['code']
        sqt = klPad.get_quotes(code)
        if not sqt:
            return False

        old_top = 0
        enable_strategies = []
        if 'strategies' in stock['strategies']:
            for k, sobj in stock['strategies']['strategies'].items():
                if sobj['key'] == 'StrategySellELS' and 'topprice' in sobj:
                    old_top = float(sobj['topprice'])
                if sobj['enabled']:
                    enable_strategies.append(sobj['key'])
        if old_top > sqt['price']*1.05:
            return False

        if len(enable_strategies) == 1 and enable_strategies[0] in self.recycle_strs:
            return False

        smeta = None
        if sqt['price'] == klPad.get_zt_price(code):
            # 涨停
            smeta = {
                'StrategySellELS':{'enabled': True, 'topprice': round(sqt['price']*1.05, 2), 'guardPrice': round(sqt['price']*0.95, 2)},
                'StrategySellBE':{'enabled': True}
            }
        elif sqt['price'] * 1.03 > old_top:
            smeta = {
                'StrategySellELS':{'enabled': True, 'topprice': round(sqt['price']*1.03, 2), 'guardPrice': round(sqt['price']*0.95, 2)},
                'StrategySellBE':{'enabled': True, 'sell_conds': 4}
            }

        if smeta:
            strategies = self.create_strategy_keep_recycles(code, smeta, stock['strategies'])
            self.save_stock_strategy(code, strategies)
            return True

    def verify_zt1bk_strategies(self, stock):
        if stock['holdCount'] <= 0:
            return
        code = stock['code']
        if 'strategies' not in stock['strategies']:
            logger.error('%s %s has no strategy %s', self.keyword, code, stock)
            return
        sqt = klPad.get_quotes(code)
        if not sqt:
            logger.error('%s %s has no quotes %s', self.keyword, code, stock)
            return
        if sqt['high'] < sqt['lclose'] * 1.09:
            return
        top = sqt['price'] * 1.05 if sqt['price'] == sqt['high'] else min(sqt['high']*1.03, sqt['price']*1.05)
        guard = sqt['price'] * 0.95
        smeta = {
            'StrategySellELS':{'enabled': True, 'topprice': round(top, 2), 'guardPrice': round(guard, 2)},
            'StrategySellBE':{'enabled': True}
        }
        strategies = self.create_strategy_keep_recycles(code, smeta, stock['strategies'])
        self.save_stock_strategy(code, strategies)

    def verify_hrszt0_strategies(self, stock):
        if self.keyword not in ('normal', 'collat'):
            return self.verify_zt1bk_strategies(stock)

        if stock['holdCount'] <= 0:
            return
        code = stock['code']
        if 'strategies' not in stock['strategies']:
            logger.error('%s %s has no strategy %s', self.keyword, code, stock)
            return
        sqt = klPad.get_quotes(code)
        if not sqt:
            logger.error('%s %s has no quotes %s', self.keyword, code, stock)
            self.save_stock_strategy(code, stock['strategies'])
            return
        if sqt['high'] < sqt['lclose'] * 1.09:
            self.save_stock_strategy(code, stock['strategies'])
            return
        top = sqt['price'] * 1.08
        smeta = {
            'StrategySellELS':{'enabled': True, 'topprice': round(top, 2)},
            'StrategySellBE':{'enabled': True, 'sell_conds': 1 if sqt['price'] == sqt['high'] else 8}
        }
        strategies = self.create_strategy_keep_recycles(code, smeta, stock['strategies'])
        self.save_stock_strategy(code, strategies)

    def verify_hotstks_open_strategies(self, stock):
        return self.verify_hrszt0_strategies(stock)

    def verify_zt1wb_strategies(self, stock):
        if stock['holdCount'] <= 0:
            logger.info('%s %s has no holdCount %s', self.keyword, stock['code'], stock)
            return
        code = stock['code']
        sqt = klPad.get_quotes(code)
        if not sqt:
            logger.error('%s %s has no quotes %s', self.keyword, code, stock)
            return
        if sqt['price'] == klPad.get_zt_price(code):
            top = sqt['price'] * 1.05
            guard = sqt['price'] * 0.95
            smeta = {
                'StrategySellELS':{'enabled': True, 'topprice': round(top, 2), 'guardPrice': round(guard, 2)},
                'StrategySellBE':{'enabled': True}
            }
        else:
            today = guang.today_date('-')
            rec = next((r for r in stock['strategies'].get('buydetail', []) if r['date'].split(' ')[0] == today), None)
            price = rec['price'] if rec else sqt['open']
            earn = sqt['price'] * 1.03 > price * 1.05
            top = sqt['price'] * 1.03 if earn else price * 1.05
            guard = sqt['price'] * 0.95 if earn else price * 0.92
            smeta = {
                'StrategySellELS':{'enabled': True, 'topprice': round(top, 2), 'guardPrice': round(guard, 2)}
            }
            if earn:
                smeta['StrategySellBE'] = {'enabled': True, 'sell_conds': 4}
        strategies = self.create_strategy_keep_recycles(code, smeta, stock['strategies'])
        self.save_stock_strategy(code, strategies)

    def verify_recycle_strategies(self, stock):
        '''
        当日清仓的票，如果策略是'StrategyGE'或'StrategyBSBE'，清理可能的中间变量
        '''
        if stock['holdCount'] > 0 or 'strategies' not in stock['strategies']:
            return
        remain_strs = []
        for k, sobj in stock['strategies']['strategies'].items():
            if sobj['key'] not in self.recycle_strs:
                continue
            remain_strs.append(k)
            if sobj['key'] == 'StrategyGE':
                smeta = sfac.get_strategy_meta(self.keyword, stock['code'], sobj['key'])
                if not smeta:
                    continue
                if 'guardPrice' in sobj:
                    del sobj['guardPrice']
                sobj['inCritical'] = False
            if sobj['key'] == 'StrategyBSBE':
                smeta = sfac.get_strategy_meta(self.keyword, stock['code'], sobj['key'])
                if smeta and 'guardPrice' in smeta:
                    sobj['guardPrice'] = smeta['guardPrice']
                    continue
                logger.info('StrategyBSBE guardPrice not set for %s', stock)
        stock['strategies']['strategies'] = {k:v for k,v in stock['strategies']['strategies'].items() if k in remain_strs}
        stock['strategies']['transfers'] = {k:v for k,v in stock['strategies']['transfers'].items() if k in remain_strs}
        self.save_stock_strategy(stock['code'], stock['strategies'])

    def verify_track_buysell(self, stock):
        '''
        模拟账户检查买卖点，9:30之前买入的按开盘价，14:57之后买入的按收盘价，当天全天一字板涨停的无法买入需舍弃
        已处理并保存策略返回True, 无法处理后续也无法处理的也返回True, 需后续操作的返回False
        '''
        if self.keyword in ('normal', 'collat', 'credit'):
            return False

        code = stock['code']
        sqt = klPad.get_quotes(code)
        if not sqt:
            return True
        buydetails = stock['strategies'].get('buydetail', [])
        buydetails_full = stock['strategies'].get('buydetail_full', [])
        if not buydetails or not buydetails_full:
            return True

        today = guang.today_date('-')
        zt_price = klPad.get_zt_price(code)
        zt1yzb = zt_price == sqt['price'] and zt_price == sqt['low']
        if zt1yzb:
            buydetails = [rec for rec in buydetails if rec['date'].split(' ')[0] != today]
            buydetails_full = [rec for rec in buydetails_full if rec['date'].split(' ')[0] != today]
            stock['strategies']['strategies'] = {}
            self.save_stock_strategy(code, stock['strategies'])
            return True

        for rec in buydetails:
            if rec['date'].split(' ')[0] == today:
                if rec['date'] < today + ' ' + '09:30':
                    rec['price'] = sqt['open']
                elif rec['date'] > today + ' ' + '14:57':
                    rec['price'] = sqt['price']
                rec['date'] = today
        for rec in buydetails_full:
            if rec['date'].split(' ')[0] == today:
                if rec['date'] < today + ' ' + '09:30':
                    rec['price'] = sqt['open']
                elif rec['date'] > today + ' ' + '14:57':
                    rec['price'] = sqt['price']
                rec['date'] = today
        return False

    def save_stock_strategy(self, code, strategy):
        logger.info('set strategy for %s %s', self.keyword, strategy)
        url = guang.join_url(accld.dserver, 'stock')
        data = {
            'act': 'strategy',
            'acc': self.keyword,
            'code': get_fullcode(code).upper(),
            'data': json.dumps(strategy)
        }

        guang.post_data(url, data, accld.headers)


class accld:
    dserver = None
    headers = None
    all_accounts: dict[str, Account] = {}

    @classmethod
    def load_accounts(self):
        url = f"{self.dserver}userbind?onlystock=1"
        accs = guang.get_request_json(url, self.headers)
        try:
            accs = [{'name': 'normal', 'email': '', 'realcash': 1}] + [x for x in accs if x['name'] == 'collat' or x['realcash'] == 0]
        except Exception as e:
            accs = [{**x, 'name': x['username'].split('.')[1]} for x in accs]
        for acc in accs:
            account = Account(acc['name'])
            account.load_watchings()
            self.all_accounts[acc['name']] = account

    @classmethod
    def cache_stock_data(self, acc, code, data):
        if not acc in self.all_accounts:
            self.all_accounts[acc] = Account(acc)
        sfac.cache_stock_strategy(acc, code, data)

    @classmethod
    def get_account_holdcount(cls, acc, code):
        if acc == '':
            acc = 'normal' if code in cls.all_accounts['normal'].stocks else 'collat'
        if acc == 'credit':
            acc = 'collat'
        if code in cls.all_accounts[acc].stocks:
            return cls.all_accounts[acc].stocks[code].get('holdCount', 0)
        return 0

    @classmethod
    def all_stocks_cached(self):
        return sum([list(acc.stocks.keys()) for acc in self.all_accounts.values()], [])

    @classmethod
    def verify_strategies(self):
        # 收盘后根据今日成交设置买入策略
        for acc in self.all_accounts.values():
            try:
                acc.verify_strategies()
            except Exception as e:
                import traceback
                logger.error(f'Error verifying strategies for {acc.keyword}: {e}')
                logger.error(traceback.format_exc())

    @classmethod
    def add_trading_remarks(self, acc, code, remark):
        hacc = acc
        if acc == 'credit':
            hacc = 'collat'
        if hacc not in self.all_accounts:
            logger.debug('add_trading_remarks Account %s not found', hacc)
            return
        self.all_accounts[hacc].trading_remarks[code] = remark

    @classmethod
    def add_stock_strategy(cls, acc, code, strategy):
        if not isinstance(strategy, dict):
            return None

        accld.cache_stock_data(acc, code, {'strategies': strategy})
        for sobj in strategy['strategies'].values():
            if not sobj['enabled']:
                continue
            s = sfac.get_strategy(sobj['key'])
            if s:
                s.add_stock(acc, code)

    @classmethod
    def disable_stock_strategy(cls, acc, code, skey):
        if not skey:
            return

        smeta = sfac.get_strategy_meta(acc, code, skey)
        if smeta and smeta['enabled']:
            smeta['enabled'] = False
            sfac.update_strategy_meta(acc, code, skey, smeta)
        else:
            return

        s = sfac.get_strategy(smeta['key'])
        if s:
            s.remove_stock(acc, code)
        logger.info(f'stock  {acc} {code} {skey} disabled')
