from binary_classification_probe import run_binary_classification_probe
from multi_classification_probe import run_multi_classification_probe

def plot_probe_trails(trail_fn, num_trails=10):
    total_training_loss = 0
    total_val_acc = 0
    for _ in range(num_trails):
        training_loss, val_acc = trail_fn()
        total_training_loss += training_loss
        total_val_acc += val_acc
    # Text for now..
    print("Average training loss = %f, Average validation accuracy = %f" % (total_training_loss / num_trails, total_val_acc / num_trails))


if __name__ == "__main__":
    '''
    Plotting binary classification probe:
    '''
    '''
    for n in range(11):
        trail_fn = lambda: run_binary_classification_probe(n, verbose=False)
        print("For binary probe with n = %d:" % n)
        plot_probe_trails(trail_fn)
    '''

    '''
    Plotting multi classification probe:
    '''
    for n in range(8):
        trail_fn = lambda: run_multi_classification_probe(n, verbose=False)
        print("For classification probe with n = %d:" % n)
        plot_probe_trails(trail_fn)