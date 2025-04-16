import pandas as pd
from os import path
import matplotlib.pyplot as plt
from matplotlib import ticker

SECONDS_PER_DAY = 24*3600

def _check_kind(kind):
    allowed_kinds = ['rewards', 'obs']
    if kind not in allowed_kinds:
        raise AttributeError(f'kind must be one of {allowed_kinds}, but was {kind}')

def dataframes_from_models(results_dir, model_nrs, kind):
    _check_kind(kind)
    data_files = [
        path.join(results_dir, f'{kind}_{model_nr}.csv')
        for model_nr in model_nrs
    ] 
    dfs = {
        f'model_{model_nr}': pd.read_csv(data_file, index_col='Time')
        for (model_nr, data_file) in zip(model_nrs, data_files)
    }
    return dfs

def aggregate_multiple_trials(log_dir, n_trials, kind):
    _check_kind(kind)
    data_files = [
        path.join(log_dir, f'rep_0{i}', f'{kind}.csv') for i in range(n_trials)
    ]
    dfs = [
        pd.read_csv(data_file, index_col='Time') for data_file in data_files
    ]
    combined_df = pd.concat(dfs, keys=range(len(dfs)))
    mean_df = combined_df.groupby(level=1).mean()
    std_df = combined_df.groupby(level=1).std()
    lower_std_df = mean_df - std_df
    upper_std_df = mean_df + std_df
    return mean_df, lower_std_df, upper_std_df

def get_plot_settings(model_name, new_labels):
    label = model_name if new_labels is None else new_labels[model_name]
    alpha = 0.5 if 'std' in model_name else 1
    linestyle = '--' if 'std' in model_name else '-'
    return label, alpha, linestyle

def compare_energyconsumption(model_dfs, kind, output_file=None,
        new_labels=None, convert_time_to_days=True):
    _check_kind(kind)
    fig, ax = plt.subplots()
    if kind=='rewards':
        for model_name, df in model_dfs.items():
            label, alpha, linestyle = get_plot_settings(model_name, new_labels)
            df['pump_price_obj'].plot(
                ax=ax, label=label, alpha=alpha, linestyle=linestyle
            )
            print(
                f'Average pump price objective for {model_name}: '
                f'{df.pump_price_obj.mean():.4f}'
            )
            plt.ylabel('Cost Efficiency')
    elif kind=='obs':
        for model_name, df in model_dfs.items():
            ec_columns = [c for c in df.columns if 'energyconsumption' in c]
            label, alpha, linestyle = get_plot_settings(model_name, new_labels)
            df.loc[:, ec_columns].sum(axis=1).plot(
                ax=ax, label=label, alpha=alpha, linestyle=linestyle
            )
            print(
                f'Average energy consumption {model_name}: '
                f'{df.loc[:, ec_columns].sum(axis=1).mean():.4f}'
            )
            plt.ylabel('Average Energy Consumption in kWh')
    if convert_time_to_days:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(SECONDS_PER_DAY))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f'{int(x/SECONDS_PER_DAY)}')
        )
    plt.legend()
    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()

if __name__=='__main__':
    results_dir = '../../Results/Multi_Network/LeakDB'
    exp_dir = 'model_4_xt'
    log_dir = path.join(results_dir, exp_dir, 'log')
    kind = 'obs'
    means, lower_stds, upper_stds = aggregate_multiple_trials(
        log_dir, n_trials=5, kind=kind
    )
    results_from_constant = pd.read_csv(
        path.join(results_dir, f'{kind}_constant_06.csv'),
        index_col='Time'
    )
    model_dfs = {
        'our model': means,
        'lower std': lower_stds,
        'upper std': upper_stds,
        'baseline': results_from_constant
    }
    output_file = (
        f'../../Results/Multi_Network/LeakDB/'
        f'comparison_ec_obs_model_4_xt_constant_06.png'
    )
    compare_energyconsumption(
        model_dfs, kind,
        output_file=output_file
    )
    plt.show()

