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
pip install anthropic google-genai
export ANTHROPIC_API_KEY=sk-ant-...      # 從 https://console.anthropic.com 取得（穿搭建議、讀尺寸表）
export GEMINI_API_KEY=...                # 從 https://aistudio.google.com 取得（試穿照）
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

### 手機單機版（`outfit/standalone/`）

同樣的身體尺寸計算，做成不需要伺服器的單一網頁：計算在瀏覽器裡跑（`measure.js` 是 `body.py`
的 JavaScript 版本，`tests/test_body.py` 會用 node 檢查兩者結果一致），MediaPipe 的程式、WASM 和模型
跟網頁放在同一個資料夾。

```sh
outfit/standalone/fetch_assets.sh   # 下載 MediaPipe 檔案（約 22 MB，不進 git）
python3 -m http.server -d outfit/standalone 8000
```

## 尺寸推薦（「📏 尺寸推薦」分頁）

1. 量一件自己穿起來最合身的衣服（平放、公分）：上衣量胸寬／肩寬／衣長／袖長，褲子量腰寬／臀寬／大腿寬／褲長。
   數字只存在使用者自己的瀏覽器（localStorage）。
2. 上傳商品尺寸表截圖，Claude 只負責把表格讀成數字（英吋換公分、圍度換成平量寬度），
   讀到的表格會顯示出來，讀錯可以直接改。
3. 推薦尺寸由 `outfit/sizing.py` 用固定公式計算，不交給 AI 猜：逐項比較商品和合身衣服的差距，
   胸寬／腰寬權重較高，偏緊比偏寬鬆扣分多（彈性布料除外），±1.5 cm 內算「剛好」。

限制：只支援商品平量尺寸表；「適合胸圍 88–92」這類身體尺寸表會請使用者改找平量尺寸表。

## 試穿（「👕 試穿」分頁）

上傳全身照，用文字描述想穿的衣服或上傳最多 3 張商品照，由 Gemini 圖片模型把照片裡的人換上那套衣服。
「👗 穿搭建議」的每套建議也有「👕 試穿這套」按鈕，會直接帶入照片和描述。

- `outfit/tryon.py`：要求模型只換衣服，臉、身形、姿勢、背景保持不變。
- 圖片模型會重畫整張圖，臉常常會跑掉，所以產生後由瀏覽器把**原本的臉和頭髮貼回去**
  （`outfit/static/keepface.js`）：MediaPipe 姿勢模型找出頭部、自拍分割模型找出臉和頭髮，
  依頭肩關節點對齊後，用柔和邊緣把原圖的頭貼在新圖上。結果頁可以切換「保留原本的臉／AI 原圖」。
  構圖或姿勢被改太多時會略過並說明原因。也會把原圖的長寬比傳給 Gemini，讓構圖一致比較好對齊。
- 模型預設 `gemini-3.1-flash-image`，可用環境變數 `OUTFIT_TRYON_MODEL` 換成其他 Gemini 圖片模型。
- 也可以改用專門做試穿的 **Genlook API**（`outfit/genlook.py`）：設定 `GENLOOK_API_KEY`（到
  platform.genlook.app 申請，註冊送 10 點）就會自動改用 Genlook；`OUTFIT_TRYON_PROVIDER=gemini|genlook`
  可以強制指定。Genlook 需要衣服的商品照（每張一件，最多 4 件一起穿），每次 1 點（約 US$0.04）。
  設 `GENLOOK_MOCK=1` 會用 Genlook 的免費測試模式，不扣點，回傳原照片，用來確認整個流程接得通。
- 圖片生成需要 Gemini 付費方案；照片會傳到 Google。
- 結果是示意圖：模型不知道衣服實際尺寸，會畫成剛好合身。買哪一號請看尺寸推薦。

## 分身與模板（「🧸 分身」分頁）

拍一張照片後：

1. **選一個模板試看**：6 個模板（城市街拍、浪漫約會、質感通勤、海島度假、韓系校園、冬日聖誕），
   都用「3D 遊戲風」（遊戲 CG 質感、正常人體比例），換上模板的穿搭和場景，一次產生一張。
2. 或**比較三種畫風**：3D 遊戲風、療癒 Q 版公仔、時尚換裝風，保留照片裡的衣服。
3. 選一張當分身存起來，之後上傳衣服照片幫分身換裝。

- `outfit/avatar.py`：模板和畫風都在這裡定義；每種都要求保留本人的身形、膚色、臉型和髮型，不修瘦、不美白。
- 半身或自拍照會自動放到加高的畫布上，讓 AI 畫出全身（臉會比較不像，全身照效果最好）。
- 產圖服務：設了 `HF_TOKEN` 就用 Hugging Face 上 **Qwen-Image-Edit-2511** 的官方免費 Demo（要排隊、
  每天有次數上限，用完會提示），否則用 Gemini（`GEMINI_API_KEY`）；`OUTFIT_AVATAR_PROVIDER=qwen|flux|gemini`
  可以指定。Qwen 一次可以看 1–3 張圖，所以分身換裝一次最多 2 件衣服；FLUX.1 Kontext 只能看 1 張，
  選 FLUX 時換裝改用 Gemini。
- 選好的分身存在跑 App 的電腦：`~/.outfit/`（可用 `OUTFIT_DATA_DIR` 改位置）。
- 不開網頁也能試：`python3 -m outfit.avatar 我的照片.jpg -o 輸出/ --template all`（或 `--style all`）。

## 示範版（可以直接在手機點的連結）

`python3 -m outfit.demo.build_demo outfit-demo.html` 會把真正的網頁加上 `outfit/demo/demo_backend.js`，
在瀏覽器裡用範例資料回應所有 API，產生一個不需要伺服器的單一頁面，可以發布成連結在手機上看畫面和流程。
AI 結果都是標示「示範」的範例；身體尺寸分頁改成連到「照片量身」單機版。

## 3D 分身（實驗）

流程：Qwen-Image-Edit 把分身畫成全身 A 字站姿 → TRELLIS.2（Hugging Face 免費 Demo）轉成 GLB 3D 模型 →
`python3 -m outfit.demo.build_viewer 模型.glb 分身.html` 產生單一網頁，手機上可以旋轉、縮放；
模型綁好骨架、帶有動作後，同一個網頁會自動出現切換動作的按鈕。
