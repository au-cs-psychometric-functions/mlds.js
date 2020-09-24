import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def mlds(filename):
    data = pd.read_table(filename, sep='\t')
    res = smf.glm('resp ~ S2 + S3 + S4 + S5 + S6 + S7 + S8 + S9 + S10 + S10 + S11 - 1', family=sm.families.Binomial(sm.families.links.probit()), data=data).fit()
    print(res.summary())

if __name__ == '__main__':
    mlds('data.txt')