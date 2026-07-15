"""Fix part 2/2: PITTopkDropoutStrategy -- a TopkDropout that actually enforces
point-in-time index membership.

WHAT WAS BROKEN in qlib's TopkDropoutStrategy:
  1. Sells are capped at n_drop per rebalance AND are chosen only from the bottom of the
     SCORED holdings. A stock removed from the index has no score at all, so it is not
     reliably selected for sale.
  2. The sell loop begins with
         if not self.trade_exchange.is_stock_tradable(code, ...): continue
     so once a name delists (its price bins end) it can NEVER be sold and is frozen in the
     portfolio forever -- while still being marked to its last price. That is the
     16,232 ghost-position-days.

WHAT THIS DOES:
  * every holding that is not an index member today is force-sold, bypassing the n_drop cap
  * the buy candidate pool is filtered to current members, so a non-member is never bought
  * normal topk-dropout behaviour is preserved for the members
Combined with pit13_fix_store.py (which appends a liquidation bar for same-day delistings),
every exit is executable and ghosts go to zero.
"""
import warnings; warnings.filterwarnings("ignore")
import copy
import numpy as np
import pandas as pd
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.position import Position
from qlib.contrib.strategy import TopkDropoutStrategy


class PITTopkDropoutStrategy(TopkDropoutStrategy):
    def __init__(self, *, membership: pd.DataFrame, **kwargs):
        """membership: bool DataFrame, index=datetime, columns=ticker (True == index member)"""
        super().__init__(**kwargs)
        self.membership = membership.sort_index()
        self.forced_sales = []      # audit trail

    def _members_on(self, ts):
        i = self.membership.index.searchsorted(pd.Timestamp(ts), side="right") - 1
        if i < 0:
            return set()
        row = self.membership.iloc[i]
        return set(row.index[row.values])

    def generate_trade_decision(self, execute_result=None):
        step = self.trade_calendar.get_trade_step()
        t_start, t_end = self.trade_calendar.get_step_time(step)
        p_start, p_end = self.trade_calendar.get_step_time(step, shift=1)

        score = self.signal.get_signal(start_time=p_start, end_time=p_end)
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        if score is None:
            score = pd.Series(dtype=float)

        members = self._members_on(t_start)
        cur: Position = copy.deepcopy(self.trade_position)
        held = list(cur.get_stock_list())
        cash = cur.get_cash()

        # ---- 1. FORCED EXITS: anything we hold that is no longer an index member ----
        stale = [c for c in held if c not in members]
        sell_orders = []
        for code in stale:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code, start_time=t_start, end_time=t_end, direction=OrderDir.SELL
            ):
                # should not happen once pit13 has added the liquidation bar; record it
                self.forced_sales.append((pd.Timestamp(t_start), code, "UNSELLABLE"))
                continue
            o = Order(stock_id=code, amount=cur.get_stock_amount(code=code),
                      start_time=t_start, end_time=t_end, direction=Order.SELL)
            if self.trade_exchange.check_order(o):
                sell_orders.append(o)
                val, cost, _ = self.trade_exchange.deal_order(o, position=cur)
                cash += val - cost
                self.forced_sales.append((pd.Timestamp(t_start), code, "SOLD"))

        # ---- 2. normal topk-dropout, restricted to CURRENT MEMBERS ----
        score = score[score.index.isin(members)].dropna()
        held = [c for c in cur.get_stock_list()]                 # post-forced-exit
        last = score.reindex(held).sort_values(ascending=False).index
        # any holding with no score (still a member but unscored) sorts to the back
        last = pd.Index(list(last) + [c for c in held if c not in score.index])

        cand = score[~score.index.isin(last)].sort_values(ascending=False).index
        n_new = self.n_drop + self.topk - len(last)
        today = list(cand[: max(n_new, 0)])

        comb = score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index
        drop = list(comb[-self.n_drop:]) if self.n_drop else []
        sell = [c for c in last if c in drop]
        buy = today[: len(sell) + self.topk - len(last)]

        for code in sell:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code, start_time=t_start, end_time=t_end, direction=OrderDir.SELL
            ):
                continue
            o = Order(stock_id=code, amount=cur.get_stock_amount(code=code),
                      start_time=t_start, end_time=t_end, direction=Order.SELL)
            if self.trade_exchange.check_order(o):
                sell_orders.append(o)
                val, cost, _ = self.trade_exchange.deal_order(o, position=cur)
                cash += val - cost

        buy_orders = []
        value = cash * self.risk_degree / len(buy) if len(buy) else 0
        for code in buy:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code, start_time=t_start, end_time=t_end, direction=OrderDir.BUY
            ):
                continue
            px = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=t_start, end_time=t_end, direction=OrderDir.BUY)
            f = self.trade_exchange.get_factor(stock_id=code, start_time=t_start, end_time=t_end)
            amt = self.trade_exchange.round_amount_by_trade_unit(value / px, f)
            buy_orders.append(Order(stock_id=code, amount=amt, start_time=t_start,
                                    end_time=t_end, direction=Order.BUY))

        return TradeDecisionWO(sell_orders + buy_orders, self)
