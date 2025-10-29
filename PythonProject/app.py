from flask import Flask, render_template_string, request
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import io, base64, os

app = Flask(__name__)

# ---------------- HTML Template ----------------
HTML = """ 
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Stock Dashboard | AI Prediction</title>
<style>
  body { font-family: 'Segoe UI', Arial; background: #0d1117; color: #e6edf3; margin: 0; }
  header { background: #161b22; padding: 20px; text-align: center; color: #58a6ff; font-size: 28px; }
  .container { max-width: 950px; margin: 40px auto; background: #161b22; padding: 30px; border-radius: 12px; box-shadow: 0 0 25px rgba(0,0,0,0.4); }
  input[type=text] { width: 70%; padding: 12px; border: none; border-radius: 6px; background: #21262d; color: #fff; }
  button { padding: 12px 20px; border: none; border-radius: 6px; background: #238636; color: white; cursor: pointer; }
  button:hover { background: #2ea043; }
  img { width: 100%; border-radius: 10px; margin-top: 25px; box-shadow: 0 0 10px rgba(0,0,0,0.6); }
  .price-box { text-align: center; margin-top: 25px; }
  .price { font-size: 48px; font-weight: bold; }
  .change { font-size: 20px; }
  .up { color: #00ff7f; }
  .down { color: #f85149; }
  .stable { color: #ccc; }
  .pred-box { margin-top: 30px; text-align: center; background: #21262d; padding: 20px; border-radius: 10px; }
  .pred { font-size: 22px; color: #ffcc00; }
  .note { text-align: center; color: #8b949e; margin-top: 15px; font-size: 13px; }
</style>
</head>
<body>
<header>📈 AI Stock Dashboard</header>
<div class="container" style="text-align:center;">
  <form method="post">
    <input type="text" name="ticker" placeholder="Enter Stock Symbol (e.g. TSLA, AAPL)" required>
    <button type="submit">Analyze</button>
  </form>

  {% if error %}
    <p style="color:#f85149;">{{ error }}</p>
  {% endif %}

  {% if name %}
  <div class="price-box">
    <h2>{{ name }} ({{ ticker }})</h2>
    <div class="price">${{ price }}</div>
    <div class="change {% if trend == 'up' %}up{% elif trend == 'down' %}down{% else %}stable{% endif %}">
      {% if trend == 'up' %}+{% endif %}{{ change }} ({{ change_percent }}%)
    </div>
  </div>

  <div class="pred-box">
    <p class="pred">🔮 Predicted Tomorrow's Close: <b>${{ predicted_price }}</b></p>
  </div>

  <img src="data:image/png;base64,{{ plot_url }}">
  <div class="note">5-Year Historical Stock Price (AI Linear Regression Prediction for Next Day)</div>
  {% endif %}
</div>
</body>
</html>
"""

# ---------------- Flask Logic ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    name = ticker = plot_url = None
    price = change = change_percent = predicted_price = 0
    trend = "stable"
    error = None

    if request.method == "POST":
        ticker = request.form["ticker"].upper()

        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5y")

            if hist.empty:
                error = f"No data found for '{ticker}'."
            else:
                latest = hist["Close"].iloc[-1]
                prev = hist["Close"].iloc[-2]
                price = round(float(latest), 2)
                change = round(float(latest - prev), 2)
                change_percent = round((change / prev) * 100, 2)

                trend = "up" if change > 0 else "down" if change < 0 else "stable"

                hist["Days"] = np.arange(len(hist))
                X = hist[["Days"]]
                y = hist["Close"]
                model = LinearRegression().fit(X, y)
                tomorrow_day = np.array([[len(hist) + 1]])
                predicted_price = round(float(model.predict(tomorrow_day)[0]), 2)

                plt.style.use("dark_background")
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(hist.index, hist["Close"], color="#00bfff", linewidth=1.8, label="Close Price")
                ax.plot(hist.index, model.predict(X), color="#ffcc00", linestyle="--", linewidth=1.2, label="Trend (LR)")
                ax.fill_between(hist.index, hist["Close"], color="#00bfff", alpha=0.1)
                ax.set_facecolor("#0d1117")
                ax.grid(color="#2f353e", linestyle="--", linewidth=0.5)
                ax.set_title(f"{ticker} - 5 Year Price Trend", color="#58a6ff", fontsize=14)
                ax.tick_params(axis="x", colors="#9ba3b0")
                ax.tick_params(axis="y", colors="#9ba3b0")
                ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=150)
                buf.seek(0)
                plot_url = base64.b64encode(buf.getvalue()).decode()
                buf.close()
                plt.close()

                name = stock.info.get("shortName", ticker)

        except Exception as e:
            error = f"⚠️ Error: {str(e)}"

    return render_template_string(HTML, name=name, ticker=ticker, price=price,
                                  change=change, change_percent=change_percent,
                                  trend=trend, predicted_price=predicted_price,
                                  plot_url=plot_url, error=error)

# ✅ Important fix for Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
