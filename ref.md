以下是整理後的 **Markdown 技術備忘稿（Developer 參考版）**，你可以直接貼進文件或 repo：

---

# AI 字幕工作流技術備忘錄（Developer 版）

## 目標

* 🎬 影片（英文）
* → 🎧 語音辨識（ASR）
* → 🈶 中文字幕
* → 📄 SRT / VTT
* ⚙️ 可自動化、穩定、高品質

---

## 🧠 核心流程（推薦 Pipeline）

```text
Video
  ↓ (可選) Extract Audio
Audio
  ↓ VAD (Voice Activity Detection)
Speech Segments
  ↓ ASR (Whisper / Faster-Whisper)
English Subtitles
  ↓ Translation (AI / NMT)
Chinese Subtitles
  ↓ Post-processing
SRT / VTT
```

---

# 1️⃣ VAD 是什麼？

## Voice Activity Detection（語音活動偵測）

### 🎯 目的

> **從整段音檔中切出「有人說話的部分」並去除靜音與噪音。**

---

## 為什麼要用 VAD？

### 👍 好處

* ✔ 提升辨識準確度
* ✔ 避免靜音造成誤判
* ✔ 加速運算
* ✔ 字幕切句更自然
* ✔ 有助多人對話處理
* ✔ 避免長音檔時間軸漂移

### 🚫 少了 VAD 可能會發生

* 長句不中斷
* 背景聲被誤成文字
* 字幕節奏不自然
* time drift

---

## 常用 VAD 工具

| 工具             | 特點          |
| -------------- | ----------- |
| **Silero VAD** | 快、輕量、穩定（推薦） |
| WebRTC VAD     | 傳統方案        |
| pyannote.audio | 支援說話者分離     |

---

## 常見參數（概念）

* `min_speech_len`：最短語音判定
* `max_pause`：最大停頓仍視為同一句
* `max_segment_len`：限制片段長度

🎯 目標：
**每句字幕 1.5–6s、16–20 中文字**

---

# 2️⃣ 是否需要先抽音訊？

## 結論

> **長影片（≥ 30–60 分鐘）→ 強烈建議先抽成純音訊再處理。**

---

## 為什麼？

### ✔ 效能更好

* 影片需要影像解碼
* 音訊更省資源

### ✔ 避免時間碼漂移

* 部分影片容器會累積誤差
* 音訊較穩定

### ✔ 更好做分段 / 續跑

* 方便 debug
* 更利於批次處理

### ✔ 格式可控

建議：

```
wav 16kHz mono PCM
```

---

## 什麼情況可不用抽音訊？

* 短片（10–30 分鐘）
* 使用 GUI 工具
* 不建 pipeline
* 機器效能不差

---

## 什麼情況一定要？

✔ ≥ 1 小時
✔ 多檔自動化
✔ 背景音樂/噪音多
✔ 需要 VAD
✔ 要最高穩定度

---

# 3️⃣ ASR 模型建議

## 首選

### **Faster-Whisper**

* 支援 CPU / GPU
* 快且省記憶體
* 安裝友善

### Python 範例

```python
from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.wav", beam_size=5)
```

CPU 模式：

```python
model = WhisperModel("small", device="cpu", compute_type="int8")
```

---

## 模型選擇建議

| 模型         | 速度     | 準確度 | 建議用途 |
| ---------- | ------ | --- | ---- |
| base       | 很快     | 中   | 草稿   |
| small      | 快      | 中高  | 一般用途 |
| **medium** | 中      | 高   | ⭐ 推薦 |
| large-v3   | 慢/需GPU | 最高  | 專業級  |

---

# 4️⃣ 翻譯策略

## 推薦流程

> **先產生英文字幕 → 再逐句翻譯**

原因：

* 保留時間碼
* 易校對
* 結果穩定

---

## 翻譯工具

### 免費 / 離線

* NLLB-200
* M2M100

### 高品質

* GPT API
* DeepL

---

# 5️⃣ GPU 是否必要？

## 結論

> **不是必需，但 GPU 會更快、可用更大模型。**

---

## 速度參考（10–20 分鐘影片）

| 環境           | 模型           | 時間       |
| ------------ | ------------ | -------- |
| CPU          | small/medium | 10–40 分鐘 |
| GPU (6–8GB↑) | medium/large | 3–8 分鐘   |

---

# 6️⃣ 字幕品質建議

✔ 每行 < 20–22 中文字
✔ 每段 1.5–6 秒
✔ 盡量依語義切句
✔ 專有名詞詞典化
✔ 校對縮寫/人名

---

# 7️⃣ 建議技術 Stack

## Python 套件

```
faster-whisper
torch
pydub
numpy
silero-vad
python-srt
```

可選翻譯：

```
openai
deepl
transformers
```

---

# 8️⃣ 推薦完整流程（最佳實務）

```text
Video
 → extract audio (wav 16k mono)
 → VAD 切段
 → ASR (Faster-Whisper)
 → 字幕合併/斷句優化
 → 翻譯（逐句）
 → 輸出 SRT/VTT
 → 人工校對
```

---

# 9️⃣ 適用場景

✔ 會議記錄
✔ 訪談 / Podcast
✔ 教學影片
✔ YouTube
✔ 公司內部素材

---

# 🔐 隱私

* Whisper / VAD 可 **全離線**
* 適合 NDA 場景
* 資料不會外傳

---

# 🏁 Summary（工程師重點）

* **VAD = 切出有人說話的片段**
* **長影片 → 先抽音訊更穩**
* **Faster-Whisper = 效能最佳解**
* **medium 模型 = CP 值最高**
* **GPU = 不是必須，但更快**
* **字幕品質靠後處理＋翻譯策略**

---

如果你想要：
✅ CLI 利用範例
✅ pipeline 架構
✅ Docker 化
✅ 批次處理流程

告訴我你的：

* OS
* Python / Node 偏好
* GPU 是否具備

我可以再補完整實作建議 👍

