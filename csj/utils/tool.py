import json
import pandas as pd 
from pandas import DataFrame

def d_to_df(path: str, d: dict) -> DataFrame:
    payload = json.loads(d) if isinstance(d, str) else d

    with open(f'{path}/kline_{payload.get("Ins", [])}.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(payload))

    open_price = []
    high_price = []
    low_price = []
    close_price = []
    amount = []
    volume = []
    timestamps = []

    for item in payload.get('data', []):
        open_price.append(item['O'])
        high_price.append(item['H'])
        low_price.append(item['L'])
        close_price.append(item['C'])
        amount.append(item['A'])
        volume.append(item['V'])
        timestamps.append(pd.to_datetime(item['TiD'] + ' ' + item['T'], format='%Y%m%d %H:%M:%S'))

    return pd.DataFrame({
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close_price,
        'volume': volume,
        'amount': amount,
        'timestamps': timestamps,
    })


def plot_prediction(kline_df, pred_df, save_path='prediction_plot.png'):
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax1.set_ylabel('Close Price', fontsize=14)
    ax1.legend(loc='lower left', fontsize=12)
    ax1.grid(True)

    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=1.5)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True)

    plt.tight_layout()

    if os.environ.get("DISPLAY") is None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)