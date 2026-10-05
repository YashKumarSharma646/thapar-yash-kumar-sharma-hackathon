# Demo video script (≤ 5 minutes)

Record the screen at 1080p with the dashboard at http://localhost:8501 and a terminal.
Before recording: run `docker compose up -d`, then `docker compose restart ingestor` about 30 seconds
before you start talking, so the replay begins on camera. The whole replay takes about 10 minutes, so
part 3 shows it in progress; the stress-testing tab keeps every test, so you can pick any of them.

| Time | Screen | Say (roughly) |
|---|---|---|
| 0:00–0:30 | Deck slide 1 | "Ripple turns financial news and social media into calibrated risk signals, and stress-tests a wholesale banking book when a high-impact event breaks. Everything runs offline with one `docker compose up`." |
| 0:30–1:00 | Terminal: `docker compose ps`, then deck slide 2 | "Six services on Redis Streams: an ingestor replays July–August 2018 news and tweets, the NLP engine turns each item into a structured RiskSignal, Module B subscribes to those signals, and FastAPI and Streamlit serve the results." |
| 1:00–2:00 | Dashboard, **Risk signals** tab | Point at the replay clock and the noise-filter KPI: "91% of company tweets are spam, filtered out." Show the sentiment chart for FB around 25–26 July: "Facebook's record earnings crash: sentiment drops and attention spikes." Open the top alert's evidence card: "Event type, sentiment and severity come from a 33M model distilled from Qwen 7B. Impact is the probability of a 2-sigma price move, calibrated on 2014–2020 history, and the card shows what drove it." |
| 2:00–3:30 | **Stress testing** tab | "Each high-impact signal triggers a stress test." Pick the FB earnings test: "a company event shocks Facebook's equity and credit: a few million dollars." Pick a Turkey or trade-war test: "a market-wide event replays a real historical episode, scaled by impact. Here's the P&L by asset class and region, the largest hits, and the CET1 ratio before and after under Basel IRB." Open *Shocks applied and assumptions*. |
| 3:30–4:15 | **What-if** tab | Set Entity = MARKET, Event = Macroeconomic, Impact = 10: "the taper-tantrum replay. Bonds lose, the pay-fixed swap hedges." Switch to TURKEY + Credit Event: "the lira crisis. The short-lira forward offsets the FX loss on lira loans." |
| 4:15–5:00 | Deck slides 3–4 and 7 | "Results: event F1 0.57 to 0.75 versus keyword rules; alerts are followed by a real 2-sigma move about four times more often than chance. Limitations: teacher labels rather than human labels, a synthetic book, first-order pricing. Next: a live feed, reverse stress tests, and Module A." |

Tips: keep the browser zoom at 90% so a whole tab fits; close other tabs; speak over the replay rather than waiting for it.
