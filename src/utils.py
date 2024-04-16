from matplotlib import pyplot as plt
import numpy as np

def plot_graphs(history, title):
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))

    # Accuracy plot
    train_acc = [item['train_acc'] for item in history]
    test_acc = [item['test_acc'] for item in history]
    axs[0, 0].plot(train_acc, label='Train accuracy', color='blue')
    axs[0, 0].plot(test_acc, label='Test accuracy', color='orange')
    axs[0, 0].set_title('Accuracy')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Accuracy')
    axs[0, 0].set_ylim(0.65, 1.05)
    axs[0, 0].grid(True)
    axs[0, 0].legend(loc='best')

    # Loss plot
    train_loss = [item['train_loss'] for item in history]
    test_loss = [item['test_loss'] for item in history]
    axs[0, 1].plot(train_loss, label='Train loss', color='blue')
    axs[0, 1].plot(test_loss, label='Test loss', color='orange')
    axs[0, 1].set_title('Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].grid(True)
    axs[0, 1].legend(loc='best')

    # Learning rate plot
    lrs = [item['lrs'][-1] for item in history]
    axs[1, 0].plot(lrs, label='Learning rate', color='green')
    axs[1, 0].set_title('Learning Rate')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Learning Rate')
    axs[1, 0].grid(True)
    axs[1, 0].legend(loc='best')
    axs[1, 0].text(len(lrs) - 1, lrs[-1], f"Final LR: {lrs[-1]:.7f}", ha='right', va='bottom', fontsize=8)

    # Time per epoch plot
    time_per_epoch = [item['time_per_epoch'] for item in history]
    time_per_epoch = time_per_epoch[0]
    axs[1, 1].plot(time_per_epoch, label='Time per epoch', color='red')
    axs[1, 1].set_title('Time per Epoch')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Time (s)')
    axs[1, 1].grid(True)
    axs[1, 1].legend(loc='best')

    fig.suptitle(title)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()