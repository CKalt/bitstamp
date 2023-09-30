import json

class DataFeeder:
    def get_data(self, mode, file_paths):
        if mode == "playback":
            data_streams = {}
            for pair, file_path in file_paths.items():
                with open(file_path, "r") as file:
                    lines = file.readlines()
                    print(f"Read {len(lines)} lines from {file_path}")
                data_streams[pair] = (json.loads(line) for line in lines)
            return data_streams
