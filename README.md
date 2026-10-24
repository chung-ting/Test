# FP&A toolkit + 台股短期趨勢預測

This repo holds two independent packages: `fpa/` (below) and `twpred/` (台股預測，見文末).

Dependency-free Python (3.10+) tools for monthly financial planning & analysis:

- **Budget vs. actual variance** per account and period, with favorable/unfavorable
  flagging by category (revenue over budget is good; costs over budget are bad) and
  materiality thresholds (absolute **or** percentage).
- **P&L summary**: revenue, COGS, gross profit and margin, opex, operating income.
- **Run-rate forecast**: projects revenue and costs forward using the average
  month-over-month growth of the last 3 months.
- **Cash runway** from a cash balance and the latest month's burn.

## Usage

```sh
python3 -m fpa --budget data/budget.csv --actuals data/actuals.csv --cash 2500000
python3 -m fpa --budget data/budget.csv --actuals data/actuals.csv \
  --abs-threshold 25000 --pct-threshold 5 --forecast-months 12 -o report.md
```

## Input format

CSV with columns `period,account,category,amount`:

| column   | notes                                   |
|----------|-----------------------------------------|
| period   | `YYYY-MM`                               |
| account  | any name; must keep one category        |
| category | `revenue`, `cogs`, or `opex`            |
| amount   | positive number; commas allowed         |

`data/` has a sample quarter (Jul–Sep 2026) for a SaaS business.

## Tests

```sh
python3 -m unittest -v
```

---

# twpred：台股短期趨勢預測 MVP

結合 **技術面、籌碼／風向、月營收**，用 LightGBM 預測「未來 5 天扣成本後報酬 > 2%」的機率，
並用 walk-forward 回測，誠實地呈現「出手門檻越嚴 → 勝率越高 → 機會越少」這條曲線。

## 安裝與執行

```sh
pip install -r requirements.txt
export FINMIND_TOKEN=你的token      # 到 https://finmindtrade.com 免費註冊，可不設但請求上限較低

# 回測（第一次會下載台灣 50 的資料並快取到 data/cache/，約 200 次 API 請求）
python3 -m twpred backtest --start 2017-01-01 --end 2026-10-23

# 今天收盤後的訊號：前 10 名
python3 -m twpred signal --start 2021-01-01 --end 2026-10-23 --top 10
```

常用參數：`--stocks 2330,2317,2454`（自選股票）、`--horizon 5`（持有天數）、
`--target 0.02`（標籤門檻）、`--fee-discount 0.28`（券商手續費折數）、
`--top 3 --min-prob 0.5`（投資組合模擬的出手規則）。

報告輸出在 `reports/report.md`，另有 `predictions.csv`（每筆樣本外預測）與 `equity_curve.csv`。

## 報告怎麼看

1. **門檻 vs. 勝率表**：先看第一列「全部（基準）」的勝率，模型挑出來的要明顯比它高才有用。
   看 **95% 信賴區間的下限**，交易次數少時高勝率可能只是運氣。
2. **AUC**：0.5 是瞎猜；短線股價能穩定在 0.53～0.58 就算有東西。
3. **逐年表現**：每一年都要比基準好，只有某一年好多半是運氣。
4. **投資組合**：看扣成本後的報酬、最大回撤，並和加權指數比。

## 防止回測作弊的設計

| 問題 | 處理方式 |
|---|---|
| 用到未來資料 | 所有特徵只用 t 日收盤前已公開的資訊；月營收視為次月 11 日才可用 |
| 標籤重疊 | walk-forward 訓練集剔除最後 horizon 天（purge） |
| 進場價不實 | t+1 **開盤**進場；隔天一字漲停視為買不到 |
| 忽略成本 | 手續費 0.1425%×2 + 證交稅 0.3% 全部扣除 |
| 參數過度擬合 | LightGBM 參數固定，不對回測結果調參 |
| 門檻偷看 | 只用事前可知的門檻（每日前 K 名、固定機率） |

測試（`python3 -m unittest`）用合成資料驗證：純雜訊資料 AUC 必須約 0.5，
截斷未來資料後特徵不能改變，埋入的已知訊號要能被找出來。

## 已知限制 / 下一步

- 股票池是「現在」的台灣 50，有倖存者偏差 → 改用歷史成分股或全市場
- 未含隔夜美股、費半、台積電 ADR、台指期外資未平倉 → 第 3 階段
- 未含新聞、PTT 情緒 → 第 4 階段
- 若 FinMind 帳號不支援 `TaiwanStockPriceAdj`，會退回未還原價，除權息日會有假跌幅

本工具僅供研究，不構成投資建議。

---

# outfit — 穿搭分析網頁

拍一張全身照，選擇場合（日常、上班、約會…），由 Claude 分析目前穿搭（評分、優點、可改進處）、
適合的剪裁與顏色（附色票），並提出 3 套完整穿搭建議。資料來源就是使用者拍的照片，不需要外部資料庫。

```sh
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...      # 從 https://console.anthropic.com 取得
python3 -m outfit                        # 開 http://127.0.0.1:8000
python3 -m outfit --host 0.0.0.0         # 讓同一個 Wi-Fi 的手機用 http://<電腦IP>:8000 開啟
```

- 照片在瀏覽器先縮到最長邊 1568px 再上傳，伺服器不儲存照片。
- 每次分析約 NT$0.5–1（Claude API 計費）。
- 拒答時會自動改由其他模型回答（server-side fallback）。

## 身體尺寸（「🧍 身體尺寸」分頁）

輸入身高，拍正面、側面兩張貼身衣服的全身照，推算肩寬、胸圍、腰圍、臀圍、內側腿長、手臂長。

- 照片只在手機瀏覽器裡用 MediaPipe（pose landmarker + 人體分割）分析，**不上傳**；
  只把關節點座標和每一列的人體輪廓範圍送到伺服器。
- `outfit/body.py`：用身高換算每像素公分數，正面量寬、側面量厚，圍度用橢圓周長公式近似；
  臀圍只在胯下以上找最寬處，避免把張開的大腿算進去。
- 準確度大約圍度 ±3–5 cm、長度 ±2 cm。用軟尺量一次胸／腰／臀圍並按「校正」，
  之後會自動乘上個人校正係數（只存在瀏覽器）。
- 姿勢不標準、看不到肩膀臀部、頭腳被切到、正側面拍反等情況會直接提示重拍。

## 尺寸推薦（「📏 尺寸推薦」分頁）

1. 量一件自己穿起來最合身的衣服（平放、公分）：上衣量胸寬／肩寬／衣長／袖長，褲子量腰寬／臀寬／大腿寬／褲長。
   數字只存在使用者自己的瀏覽器（localStorage）。
2. 上傳商品尺寸表截圖，Claude 只負責把表格讀成數字（英吋換公分、圍度換成平量寬度），
   讀到的表格會顯示出來，讀錯可以直接改。
3. 推薦尺寸由 `outfit/sizing.py` 用固定公式計算，不交給 AI 猜：逐項比較商品和合身衣服的差距，
   胸寬／腰寬權重較高，偏緊比偏寬鬆扣分多（彈性布料除外），±1.5 cm 內算「剛好」。

限制：只支援商品平量尺寸表；「適合胸圍 88–92」這類身體尺寸表會請使用者改找平量尺寸表。
