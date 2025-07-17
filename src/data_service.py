#!/usr/bin/env python3
"""
Standalone data service that keeps historical data in memory
Allows main trading server to restart without reloading data
"""

import zmq
import pandas as pd
import pickle
import json
import logging
import threading
import time
from datetime import datetime
import psutil
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataService")

class DataService:
    def __init__(self, port=5555):
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")
        
        self.data_store = {}  # symbol -> dataframe
        self.metadata = {}    # symbol -> metadata
        self.running = True
        
        logger.info(f"Data service started on port {port}")
        
    def load_data(self, symbol, file_path):
        """Load data from file into memory"""
        logger.info(f"Loading {symbol} from {file_path}")
        start_time = time.time()
        
        # This would be your actual data loading logic
        # For now, placeholder
        rows_loaded = 0
        
        # Store metadata
        self.metadata[symbol] = {
            "file_path": file_path,
            "loaded_at": datetime.now().isoformat(),
            "rows": rows_loaded,
            "load_time": time.time() - start_time
        }
        
        logger.info(f"Loaded {rows_loaded} rows in {time.time() - start_time:.1f}s")
        
    def handle_request(self, request):
        """Handle incoming data requests"""
        try:
            cmd = request.get("command")
            
            if cmd == "get_data":
                symbol = request.get("symbol")
                start_date = request.get("start_date")
                end_date = request.get("end_date")
                
                if symbol in self.data_store:
                    df = self.data_store[symbol]
                    # Filter by date range if requested
                    if start_date:
                        df = df[df.index >= start_date]
                    if end_date:
                        df = df[df.index <= end_date]
                    
                    return {
                        "success": True,
                        "data": df.to_json(),
                        "rows": len(df),
                        "metadata": self.metadata.get(symbol, {})
                    }
                else:
                    return {"success": False, "error": f"No data for {symbol}"}
                    
            elif cmd == "status":
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                return {
                    "success": True,
                    "symbols": list(self.data_store.keys()),
                    "memory_mb": memory_usage,
                    "metadata": self.metadata
                }
                
            elif cmd == "reload":
                symbol = request.get("symbol")
                file_path = request.get("file_path")
                self.load_data(symbol, file_path)
                return {"success": True, "message": f"Reloaded {symbol}"}
                
            else:
                return {"success": False, "error": f"Unknown command: {cmd}"}
                
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return {"success": False, "error": str(e)}
    
    def run(self):
        """Main service loop"""
        while self.running:
            try:
                # Wait for request
                message = self.socket.recv_json()
                logger.debug(f"Received request: {message}")
                
                # Process request
                response = self.handle_request(message)
                
                # Send response
                self.socket.send_json(response)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Service error: {e}")
                self.socket.send_json({"success": False, "error": str(e)})
        
        logger.info("Data service shutting down")
        self.socket.close()
        self.context.term()

class DataServiceClient:
    """Client for connecting to data service"""
    
    def __init__(self, host="localhost", port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        
    def get_data(self, symbol, start_date=None, end_date=None):
        """Get data from service"""
        request = {
            "command": "get_data",
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date
        }
        
        self.socket.send_json(request)
        response = self.socket.recv_json()
        
        if response.get("success"):
            # Convert JSON back to dataframe
            df = pd.read_json(response["data"])
            return df
        else:
            raise Exception(response.get("error", "Unknown error"))
    
    def get_status(self):
        """Get service status"""
        self.socket.send_json({"command": "status"})
        return self.socket.recv_json()
    
    def close(self):
        """Close client connection"""
        self.socket.close()
        self.context.term()

if __name__ == "__main__":
    # Run as standalone service
    service = DataService()
    try:
        service.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")