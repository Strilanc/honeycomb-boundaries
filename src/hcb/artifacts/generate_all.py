import pathlib

from hcb.artifacts.code_distance_table import produce_code_distance_latex_table


def main():
    out_dir = pathlib.Path(__file__).parent.parent.parent.parent / 'out'

    p = out_dir / 'code_distance_table.tex'
    print('writing', p.name, '...', end=None)
    data = produce_code_distance_latex_table()
    with open(out_dir / 'code_distance_table.tex', 'w') as f:
        print(data, file=f)
    print('done')

    print('finished generating tables and diagrams')


if __name__ == '__main__':
    main()
