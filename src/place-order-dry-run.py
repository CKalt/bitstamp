import sys
import os

def main():
    # Ensure the trades directory exists
    if not os.path.exists("trades"):
        os.makedirs("trades")

    command = "python src/place-order.py " + ' '.join(sys.argv[1:])
    with open("trades/place-order-dry-runs.log", "a") as f:
        f.write(command + '\n')

if __name__ == '__main__':
    main()
