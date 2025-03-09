/********************************************************************
 * File Path: tdrv/pages/index.js
 *
 * A simple Next.js page that:
 *   - fetches candlestick data from tdr.py's REST server
 *   - fetches MA lines if "MA" is the active strategy
 *   - draws a candle chart
 *   - polls for updates every 60 seconds
 *   - allows the user to switch time frames
 *
 * PLEASE ensure your Python server is running at localhost:5000
 ********************************************************************/
import React, { useEffect, useState, useCallback } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  CandlestickController,
  CandleController,
  TimeScale
} from 'chart.js';
import { Chart } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  CandlestickController,
  CandleController,
  TimeScale,
  Tooltip,
  Legend
);

export default function HomePage() {
  const [candles, setCandles] = useState([]);
  const [timeframe, setTimeframe] = useState('1h');
  const [maData, setMaData] = useState({ short_ma: [], long_ma: [] });
  const [strategy, setStrategy] = useState(null);

  const fetchData = useCallback(async () => {
    try {
      // 1) Fetch the current strategy
      const stratRes = await fetch(`http://127.0.0.1:5000/api/strategy`);
      const stratJson = await stratRes.json();
      setStrategy(stratJson);

      // 2) Fetch candles
      const candleRes = await fetch(`http://127.0.0.1:5000/api/candles?symbol=btcusd&timeframe=${timeframe}`);
      const candleJson = await candleRes.json();
      setCandles(candleJson);

      // 3) If strategy is MA, fetch the short/long MA data
      if (stratJson.strategy === "MA") {
        const maRes = await fetch(`http://127.0.0.1:5000/api/indicators/ma?symbol=btcusd&timeframe=${timeframe}`);
        const maJson = await maRes.json();
        setMaData(maJson);
      } else {
        setMaData({ short_ma: [], long_ma: [] });
      }
    } catch (err) {
      console.error("Error fetching data", err);
    }
  }, [timeframe]);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60000); // poll every minute
    return () => clearInterval(interval);
  }, [fetchData]);

  const handleChangeTimeframe = (e) => {
    setTimeframe(e.target.value);
  };

  // We transform the data into chart.js-friendly format:
  // Chart.js can do candlestick if we have each data point in the form {x: <time>, o, h, l, c}.
  const candleDataForChart = candles.map((c) => ({
    x: new Date(c.timestamp * 1000),
    o: c.open,
    h: c.high,
    l: c.low,
    c: c.close
  }));

  // We'll transform the short_ma, long_ma data into line format => {x, y}
  const shortMAData = (maData.short_ma || []).map((pt) => ({
    x: new Date(pt.timestamp * 1000),
    y: pt.Short_MA
  }));
  const longMAData = (maData.long_ma || []).map((pt) => ({
    x: new Date(pt.timestamp * 1000),
    y: pt.Long_MA
  }));

  const chartData = {
    datasets: [
      {
        label: 'BTC-USD',
        data: candleDataForChart,
        type: 'candlestick',
        yAxisID: 'y',
      },
      {
        label: 'Short MA',
        data: shortMAData,
        type: 'line',
        borderColor: 'rgba(0, 153, 255, 1)',
        backgroundColor: 'rgba(0, 153, 255, 0.1)',
        pointRadius: 0,
        borderWidth: 1,
        hidden: shortMAData.length === 0
      },
      {
        label: 'Long MA',
        data: longMAData,
        type: 'line',
        borderColor: 'rgba(255, 153, 0, 1)',
        backgroundColor: 'rgba(255, 153, 0, 0.1)',
        pointRadius: 0,
        borderWidth: 1,
        hidden: longMAData.length === 0
      },
    ]
  };

  const options = {
    responsive: true,
    scales: {
      x: {
        type: 'time',
        time: {
          tooltipFormat: 'MMM dd HH:mm',
        },
      },
      y: {
        position: 'left'
      },
    }
  };

  return (
    <div style={{ width: '95%', margin: '0 auto', padding: '1rem' }}>
      <h1>tdrv - Crypto Chart for BTCUSD</h1>

      <div style={{ marginBottom: '1rem' }}>
        <label>Timeframe: </label>
        <select value={timeframe} onChange={handleChangeTimeframe}>
          <option value="15m">15m</option>
          <option value="30m">30m</option>
          <option value="1h">1h</option>
          <option value="4h">4h</option>
          <option value="1d">1d</option>
          <option value="1w">1w</option>
        </select>
      </div>

      {strategy && strategy.strategy ? (
        <p>Active Strategy: {strategy.strategy}</p>
      ) : (
        <p>Active Strategy: None</p>
      )}

      <div style={{ height: '600px', background: '#fafafa', padding: '1rem' }}>
        <Chart data={chartData} options={options} />
      </div>
    </div>
  );
}
