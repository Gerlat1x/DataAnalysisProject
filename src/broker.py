from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class SimpleBroker:
    """
    一个最简化的回测经纪商 (Broker) 模拟器：
    - 支持现金账户、持仓账户
    - 支持按目标权重下单，撮合于当日开盘价（可加滑点与佣金）
    - 每日收盘时进行盯市，记录权益曲线
    """

    # 初始资金
    initial_cash: float = 1_000_000.0
    # 手续费（基点，bps = 0.01%）
    commission_bps: float = 0.0
    # 滑点（基点，买入加价，卖出减价）
    slippage_bps: float = 0.0

    # 以下字段由 __post_init__ 初始化
    cash: float = field(init=False)  # 当前现金余额
    positions: Dict[str, int] = field(default_factory=dict, init=False)  # 持仓 {symbol: shares}
    trades: List[Dict] = field(default_factory=list, init=False)         # 成交记录列表
    equity_curve: List[Dict] = field(default_factory=list, init=False)  # 每日权益曲线

    def __post_init__(self):
        """初始化时设置现金为初始资金。"""
        self.cash = float(self.initial_cash)

    def _execute_trades(
        self,
        date: pd.Timestamp,
        open_prices: pd.Series,
        target_weights: pd.Series,
    ):
        """
        执行交易：
        - 根据目标权重计算目标市值
        - 与当前市值比较，得出差额
        - 以开盘价（加滑点）撮合交易，更新现金和持仓
        - 记录成交明细
        """
        # 确保缺失值填 0
        target_weights = target_weights.fillna(0.0)

        # 所有需要关注的股票（目标或已有持仓）
        symbols = set(target_weights.index) | set(self.positions.keys())

        # 当前总权益 = 现金 + 持仓市值
        total_equity = self.cash
        for sym, qty in self.positions.items():
            price = open_prices.get(sym)
            if pd.isna(price) or price <= 0:
                continue
            total_equity += qty * price

        # 对每只股票计算目标仓位并下单
        for sym in symbols:
            price = open_prices.get(sym)
            if pd.isna(price) or price <= 0:
                continue

            # 目标权重
            w = target_weights.get(sym, 0.0)
            # 当前持仓股数
            current_qty = self.positions.get(sym, 0)

            # 目标市值与当前市值
            target_value = total_equity * w
            current_value = current_qty * price
            value_diff = target_value - current_value

            # 换算为股数（取整）
            qty_diff = int(value_diff // price)
            if qty_diff == 0:
                continue

            # 买入还是卖出
            side = "buy" if qty_diff > 0 else "sell"

            # 滑点调整后的成交价
            slip = self.slippage_bps / 10000.0
            trade_price = price * (1 + slip if qty_diff > 0 else 1 - slip)

            # 成交金额（买为负，卖为正）
            trade_cash = trade_price * qty_diff

            # 手续费
            commission = abs(trade_cash) * self.commission_bps / 10000.0

            # 更新现金余额
            self.cash -= trade_cash + commission

            # 更新持仓
            self.positions[sym] = current_qty + qty_diff

            # 记录成交
            self.trades.append({
                "datetime": date,
                "symbol": sym,
                "side": side,
                "price": trade_price,
                "qty": qty_diff,
                "commission": commission,
                "cash": self.cash,
            })

    def _mark_to_market(self, date: pd.Timestamp, close_prices: pd.Series):
        """
        每日收盘盯市：
        - 计算当日总权益（现金 + 持仓市值）
        - 记录到 equity_curve
        """
        equity = self.cash
        for sym, qty in self.positions.items():
            price = close_prices.get(sym)
            if pd.isna(price) or price <= 0:
                continue
            equity += qty * price

        self.equity_curve.append({
            "datetime": date,
            "equity": equity,
            "cash": self.cash,
        })

    def simulate(
        self,
        panel: pd.DataFrame,
        signals: Dict[pd.Timestamp, pd.Series]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        主循环：逐日模拟回测。

        Args:
            panel: 历史数据面板 DataFrame，包含列
                   ['datetime','symbol','open','close',...]
            signals: {datetime: pd.Series(symbol->weight)}，
                     每日的目标权重信号。

        Returns:
            trades_df: 成交记录 DataFrame
            eq_df:     权益曲线 DataFrame (index=datetime, equity, cash)
        """
        # 按日期+代码排序，保证迭代一致性
        panel = panel.sort_values(["datetime", "symbol"]).copy()

        for date in panel["datetime"].drop_duplicates():
            day = panel[panel["datetime"] == date].set_index("symbol")

            # 若当日有信号 → 执行交易
            weights = signals.get(date)
            if weights is not None and not weights.empty:
                self._execute_trades(date, day["open"], weights)

            # 收盘盯市
            self._mark_to_market(date, day["close"])

        trades_df = pd.DataFrame(self.trades)
        eq_df = pd.DataFrame(self.equity_curve).set_index("datetime")

        return trades_df, eq_df
