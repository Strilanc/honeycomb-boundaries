import pathlib
import sys

from hcb.tools.analysis.collecting import MultiStats


def main():
    if len(sys.argv) == 1:
        raise ValueError("Specify csv files to include as command line arguments.")

    csvs = []
    for path in sys.argv[1:]:
        p = pathlib.Path(path)
        if p.is_dir():
            csvs.extend(p.glob("*.csv"))
        else:
            csvs.append(p)

    all_data = MultiStats.from_recorded_data(*csvs).filter(lambda e: 'PC3' not in e.circuit_style)
    print(all_data.to_csv())


if __name__ == '__main__':
    main()
