import pandas as pd
import scipy

def analyze_correlated_mnist(df):
    df['correlate_factor'] = df['correlate_factor'].astype(str)
    ax = df.plot.scatter('correlate_factor', 'test_accuracy')
    df.groupby(['correlate_factor']).mean().reset_index().plot('correlate_factor', 'test_accuracy', ax=ax, title="Avg test acc. at 1/4 epoch at various init. correlations (Gaussian)")
    ax.get_figure().savefig('correlated_plot.svg')
    pivot = df.pivot(index='idx', columns='correlate_factor', values='test_accuracy')
    print(pivot.mean())
    print(scipy.stats.ttest_1samp(pivot['1.1'] - pivot['1.0'], popmean=0, alternative='greater'))

if __name__ == '__main__':
    df = pd.read_csv('full_correlate.csv')
    analyze_correlated_mnist(df)
