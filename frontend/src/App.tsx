import { useEffect, useState } from "react";
import Plot, { baseLayout, plotConfig } from "./components/Plot";
import { getBook, getImpact, getRun, getScenarios, getStylized } from "./api/client";

const chartStyle = { width: "100%", height: "300px" };

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="stat">
      <b>{value}</b>
      <span>{label}</span>
    </div>
  );
}

export default function App() {
  const [scenarios, setScenarios] = useState<string[]>([]);
  const [scenario, setScenario] = useState("balanced");
  const [run, setRun] = useState<any>(null);
  const [sf, setSf] = useState<any>(null);
  const [impact, setImpact] = useState<any>(null);
  const [book, setBook] = useState<any>(null);

  useEffect(() => {
    getScenarios().then((d) => setScenarios(d.scenarios));
  }, []);

  useEffect(() => {
    getRun(scenario).then(setRun);
    getStylized(scenario).then(setSf);
    getImpact(scenario).then(setImpact);
    getBook(scenario).then(setBook);
  }, [scenario]);

  const returns: number[] = [];
  if (run) {
    for (let i = 1; i < run.mids.length; i++) {
      if (run.mids[i] > 0 && run.mids[i - 1] > 0)
        returns.push(Math.log(run.mids[i] / run.mids[i - 1]));
    }
  }

  return (
    <>
      <header className="app-header">
        <div>
          <h1>Market Microstructure Simulator</h1>
          <div className="sub">Zero-intelligence + Hawkes order flow → emergent stylized facts.</div>
        </div>
        <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
          {scenarios.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
      </header>

      <div className="bar">
        {run && <Stat label="events" value={run.n_events} />}
        {run && <Stat label="trades" value={run.n_trades} />}
        {sf && <Stat label="excess kurtosis" value={sf.excess_kurtosis.toFixed(2)} />}
        {sf && <Stat label="return ACF(1)" value={sf.return_acf_lag1.toFixed(3)} />}
        {impact && <Stat label="Kyle λ" value={impact.kyle_lambda.toExponential(2)} />}
      </div>

      <div className="grid">
        <div className="card full">
          <h2>Mid-price tape</h2>
          {run && (
            <Plot
              data={[{ x: run.times, y: run.mids, type: "scatter", line: { color: "#a371f7", width: 1 } }]}
              layout={{ ...baseLayout, yaxis: { ...baseLayout.yaxis, title: "mid (ticks)" } }}
              config={plotConfig}
              style={chartStyle}
            />
          )}
        </div>

        <div className="card">
          <h2>Bid-ask spread (ticks)</h2>
          {run && (
            <Plot
              data={[{ x: run.times, y: run.spreads, type: "scatter", line: { color: "#4da3ff", width: 1 } }]}
              layout={{ ...baseLayout }}
              config={plotConfig}
              style={chartStyle}
            />
          )}
        </div>

        <div className="card">
          <h2>Order-book depth (final)</h2>
          {book && (
            <Plot
              data={[
                {
                  x: book.snapshot.bids.map((l: any) => l[0]),
                  y: book.snapshot.bids.map((l: any) => l[1]),
                  type: "bar",
                  name: "bids",
                  marker: { color: "#3fb950" },
                },
                {
                  x: book.snapshot.asks.map((l: any) => l[0]),
                  y: book.snapshot.asks.map((l: any) => l[1]),
                  type: "bar",
                  name: "asks",
                  marker: { color: "#f85149" },
                },
              ]}
              layout={{ ...baseLayout, barmode: "overlay" }}
              config={plotConfig}
              style={chartStyle}
            />
          )}
        </div>

        <div className="card">
          <h2>Return distribution (fat tails)</h2>
          {run && (
            <Plot
              data={[{ x: returns, type: "histogram", marker: { color: "#a371f7" }, nbinsx: 60 }]}
              layout={{ ...baseLayout }}
              config={plotConfig}
              style={chartStyle}
            />
          )}
        </div>

        <div className="card">
          <h2>|return| autocorrelation</h2>
          {sf && (
            <Plot
              data={[{ x: sf.lags, y: sf.abs_return_acf, type: "bar", marker: { color: "#4da3ff" } }]}
              layout={{
                ...baseLayout,
                xaxis: { ...baseLayout.xaxis, title: "lag" },
                yaxis: { ...baseLayout.yaxis, title: "ACF" },
              }}
              config={plotConfig}
              style={chartStyle}
            />
          )}
        </div>

        <div className="card full">
          <h2>Price impact (square-root law)</h2>
          {impact && (
            <Plot
              data={[
                {
                  x: impact.Q,
                  y: impact.sqrt_impact,
                  type: "scatter",
                  line: { color: "#f0a500" },
                  fill: "tozeroy",
                  fillcolor: "rgba(240,165,0,.1)",
                },
              ]}
              layout={{
                ...baseLayout,
                xaxis: { ...baseLayout.xaxis, title: "order size Q" },
                yaxis: { ...baseLayout.yaxis, title: "impact" },
              }}
              config={plotConfig}
              style={chartStyle}
            />
          )}
        </div>
      </div>
    </>
  );
}
