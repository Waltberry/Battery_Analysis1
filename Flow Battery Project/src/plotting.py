import matplotlib.pyplot as plt

def plot_cycle(times, values, y_fitted, idx):
    # Plot data and fitted model
    plt.figure()
    plt.scatter(times, values, label='Data')
    plt.plot(times, y_fitted, label='Fitted Model (Gen Exp)', color='red')
    plt.xlabel('Time (hours)')
    plt.ylabel('Ewe/mV.4')
    plt.title(f'Charging Cycle {idx+1}')
    plt.legend()
    plt.show()

def print_fitted_params(fitted_params, n_terms=3):
    # Print fitted parameters
    param_labels = ['c1'] + [f'c{i+2}' for i in range(n_terms)] + [f'b{i+2}' for i in range(n_terms)]
    for label, param in zip(param_labels, fitted_params):
        print(f'{label} = {param:.4f}')