import json
import matplotlib.pyplot as plt
import pprint as pp
import argparse
import os

def quick_show_games(output_path, metadata_file='metadata.json', base_weight='weights.00000.hdf5', every_batches=20):
    with open(os.path.join(output_path, metadata_file), 'r') as f:
        metadata = json.load(f)
    games = metadata['batch_size']
    base_win_rates = metadata['wins_per_opponent'][base_weight]
    avg_win = []
    for i in range(every_batches, len(base_win_rates)):
        if i >= every_batches:
            avg_win.append(sum(base_win_rates[i-every_batches:i])/(every_batches * games))
    plt.plot(avg_win) 
    plt.show()      
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick visualization of the training results')
    parser.add_argument("out_directory", help="Path to folder where the model params and metadata will be saved after each epoch.")
    args = parser.parse_args()
    quick_show_games(args.out_directory)