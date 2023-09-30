
import json

class DataFeeder:
    def get_data(self, mode, file_paths):
        if mode == "playback":
            data_streams = {}
            for pair, file_path in file_paths.items():
                with open(file_path, "r") as file:
                    data_streams[pair] = (json.loads(line) for line in file)
            return data_streams
