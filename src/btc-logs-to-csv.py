import json
import csv
import sys

def parse_log_to_csv(log_file_path, csv_output_path):
    with open(log_file_path, 'r') as log_file, open(csv_output_path, 'w', newline='') as csv_file:
        fieldnames = ['id', 'timestamp', 'amount', 'price', 'type', 'microtimestamp', 'buy_order_id', 'sell_order_id']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for line in log_file:
            try:
                data = json.loads(line)
                trade_data = data.get('data', {})
                # Check if trade_data is not None
                if trade_data:
                    row = {
                        'id': trade_data.get('id', ''),
                        'timestamp': trade_data.get('timestamp', ''),
                        'amount': trade_data.get('amount', ''),
                        'price': trade_data.get('price', ''),
                        'type': trade_data.get('type', ''),
                        'microtimestamp': trade_data.get('microtimestamp', ''),
                        'buy_order_id': trade_data.get('buy_order_id', ''),
                        'sell_order_id': trade_data.get('sell_order_id', '')
                    }
                    writer.writerow(row)
            except json.JSONDecodeError:
                continue

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py input_log_file_path output_csv_path")
    else:
        input_log_file = sys.argv[1]
        output_csv_file = sys.argv[2]
        parse_log_to_csv(input_log_file, output_csv_file)
