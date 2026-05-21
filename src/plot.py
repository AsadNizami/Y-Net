from config import * 
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(history):
    plt.figure(figsize=(12, 6))
    graphs = ['loss', 'dice_coefficient', 'iou', 'accuracy']
    total_graphs = len(graphs)
    
    for i in range(total_graphs):
        plt.subplot(total_graphs // 2, total_graphs // 2, i+1)
        plt.plot(history.history[graphs[i]], label=f'Training {graphs[i]}')
        plt.plot(history.history[f'val_{graphs[i]}'], label=f'Validation {graphs[i]}')
        plt.title(f'Training and Validation {graphs[i]}')
        plt.xlabel('Epoch')
        plt.ylabel(graphs[i])
        plt.legend()

    plt.tight_layout()
    plt.savefig(save_path + model_name + f'/{model_name}_epochs_{epochs}.jpg')
    # plt.show()
    plt.close()
    

def add_row(file_path, data_dict):
    data_dict['Algorithm'] = model_name
    expand_col = ['loss', 'dice_coefficient', 'iou', 'accuracy', 'val_loss', 'val_dice_coefficient', 'val_iou', 'val_accuracy']
    
    for j in range(8):
        for i in range(epochs):
            data_dict[expand_col[j] + '_epoch_' + str(i)] = data_dict[expand_col[j]][i]
    
    for i in expand_col:
        data_dict.pop(i)

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=data_dict.keys())

    df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)

    df.to_csv(file_path, index=False)
