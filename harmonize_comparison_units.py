import argparse
from pathlib import Path

import pandas as pd
from refet import Daily, calcs


def harmonize_site(df, lat, elev):
    if df['min_temp'].median() > 150:
        df['min_temp'] = df['min_temp'] - 273.15
        df['max_temp'] = df['max_temp'] - 273.15

    if df['ea'].median() > 10:
        df['ea'] = df['ea'] / 100.0

    df['tmean'] = 0.5 * (df['min_temp'] + df['max_temp'])
    df['vpd'] = df.apply(
        lambda r: float(calcs._sat_vapor_pressure(r['tmean'])[0] - r['ea']),
        axis=1,
    )

    def _calc(r):
        asce = Daily(
            tmin=r['min_temp'],
            tmax=r['max_temp'],
            ea=r['ea'],
            rs=r['rsds'] * 0.0864,
            uz=r['wind'],
            zw=10.0,
            doy=int(r['doy']),
            elev=elev,
            lat=lat,
        )
        return asce.rn[0], asce.u2[0], asce.eto()[0]

    vals = df.apply(_calc, axis=1, result_type='expand')
    vals.columns = ['rn', 'u2', 'eto']
    df[['rn', 'u2', 'eto']] = vals[['rn', 'u2', 'eto']]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--comparison-dir',
        default='/mnt/mco_nas1/dgketchum/milk/weather_station_data_processing/comparison_data',
    )
    parser.add_argument(
        '--meta',
        default='/mnt/mco_nas1/dgketchum/milk/bias_ratio_data_processing/ETo/'
                'final_milk_river_metadata_nldas_eto_bias_ratios_long_term_mean.csv',
    )
    parser.add_argument('--sites', nargs='+', default=['comt', 'bfam', 'bftm'])
    parser.add_argument('--backup-dir', default='/tmp/comparison_unit_backups')
    args = parser.parse_args()

    comparison_dir = Path(args.comparison_dir)
    backup_dir = Path(args.backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta, index_col='id')

    for site in args.sites:
        src = comparison_dir / f'{site}.csv'
        if not src.exists():
            print(f'missing: {src}')
            continue

        dst = backup_dir / src.name
        df = pd.read_csv(src)
        df.to_csv(dst, index=False)

        row = meta.loc[site]
        df = harmonize_site(df, lat=row['latitude'], elev=row['elev_m'])
        df.to_csv(src, index=False)

        print(
            f'{site}: rn_mean={df["rn"].mean():.6f}, '
            f'vpd_mean={df["vpd"].mean():.6f}, eto_mean={df["eto"].mean():.6f}'
        )


if __name__ == '__main__':
    main()
