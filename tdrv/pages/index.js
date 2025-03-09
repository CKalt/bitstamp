/********************************************************************
 * File Path: tdrv/pages/index.js
 *
 * A simple Next.js page that:
 *   - fetches candlestick data from tdr.py's REST server
 *   - fetches MA lines if "MA" is the active strategy
 *   - fetches RSI if "RSI" is the active strategy
 *   - draws a candle chart
 *   - draws an RSI chart if RSI is active
 *   - polls for updates every 60 seconds
 *   - allows the user to switch time frames
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
  /********************************************************************
   * NEW CODE: we also track RSI data
   ********************************************************************/
  const [rsiData, setRsiData] = useState([]);
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
        setRsiData([]); // Clear RSI data
      }
      // 4) If strategy is RSI, fetch the RSI data
      else if (stratJson.strategy === "RSI") {
        setMaData({ short_ma: [], long_ma: [] }); // Clear MA data
        const rsiRes = await fetch(`http://127.0.0.1:5000/api/indicators/rsi?symbol=btcusd&timeframe=${timeframe}`);
        const rsiJson = await rsiRes.json();
        setRsiData(rsiJson);
      } else {
        // For any other strategy
        setMaData({ short_ma: [], long_ma: [] });
        setRsiData([]);
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

  // Create chart data for the candlestick chart
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

  // Transform RSI data => {x, y}
  const rsiChartData = rsiData.map((pt) => ({
    x: new Date(pt.timestamp * 1000),
    y: pt.RSI
  }));

  // Candle + MA chart
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

  const chartOptions = {
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

  // If RSI is active, we display a second chart
  const rsiDataset = {
    datasets: [
      {
        label: 'RSI',
        data: rsiChartData,
        type: 'line',
        fill: false,
        borderColor: 'rgba(75, 192, 192, 1)',
        tension: 0,
        pointRadius: 0,
        borderWidth: 1
      }
    ]
  };

  const rsiOptions = {
    responsive: true,
    scales: {
      x: {
        type: 'time',
        time: {
          tooltipFormat: 'MMM dd HH:mm',
        }
      },
      y: {
        beginAtZero: true,
        max: 100,
        position: 'left'
      }
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

      <div style={{ height: '600px', background: '#fafafa', padding: '1rem', marginBottom: '2rem' }}>
        <Chart data={chartData} options={chartOptions} />
      </div>

      {strategy && strategy.strategy === "RSI" && (
        <div style={{ height: '300px', background: '#fff', padding: '1rem' }}>
          <h2>RSI</h2>
          <Chart data={rsiDataset} options={rsiOptions} />
        </div>
      )}
    </div>
  );
}
