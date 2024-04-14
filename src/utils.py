from matplotlib import pyplot as plt

def plot_graphs(history, title):
    fig, axs = plt.subplots(1, 2, figsize=(20, 6))

    train_loss = [x['train_loss'] for x in history]
    train_acc = [x['train_acc'] for x in history]
    test_loss = [x['test_loss'] for x in history]
    test_acc = [x['test_acc'] for x in history]

    axs[0].plot(train_acc, label='Train accuracy')
    axs[0].plot(test_acc, label='Test accuracy')
    axs[0].set_title('Test accuracy')
    axs[0].set_ylabel('')
    axs[0].set_xlabel('Epoch')

    axs[1].plot(train_loss, label='train loss')
    axs[1].plot(test_loss, label='test loss')
    axs[1].set_title('Loss')
    axs[1].set_ylabel('')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc="best")

    fig.suptitle(title)
    plt.show()