import math
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import statsmodels
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as sm_mod
import statsmodels.graphics.gofplots as plots
import matplotlib.pyplot as plt
import statsmodels.tools.tools as smtools
import sklearn.metrics as skm
from sklearn.model_selection import train_test_split
from factor_analyzer import FactorAnalyzer
from factor_analyzer import (ConfirmatoryFactorAnalyzer, ModelSpecificationParser)
from factor_analyzer.utils import (corr, impute_values, partial_correlations, smc)

data1 = pd.read_csv("https://donatello-telesca.squarespace.com/s/Exposure-t4yx.csv")

# Perform Factor Analysis
fa = FactorAnalyzer()
# fa.set_params(n_factors=6,rotation=None)
fa.set_params(n_factors=6,rotation='varimax')
fa.fit(data1)
# Check factors
factor_loadings = fa.loadings_
eigen_values, vectors = fa.get_eigenvalues()
communalities = fa.get_communalities()
# Create scree plot 
# plt.scatter(range(1,29),eigen_values)
# plt.plot(range(1,29),eigen_values)
# plt.title('Scree Plot')
# plt.xlabel('Factors')
# plt.ylabel('Eigenvalue')
# plt.grid()
# plt.show()

def clump_factor_vars(factor_loadings,factor_num):
	observed_vars = []
	for each in range(len(factor_loadings)):
		if factor_loadings[each].argmax() == factor_num:
			observed_vars.append(each)
	return observed_vars

factor1 = data1.iloc[:,clump_factor_vars(factor_loadings,0)]
factor2 = data1.iloc[:,clump_factor_vars(factor_loadings,1)]
factor3 = data1.iloc[:,clump_factor_vars(factor_loadings,2)]
factor4 = data1.iloc[:,clump_factor_vars(factor_loadings,3)]
factor5 = data1.iloc[:,clump_factor_vars(factor_loadings,4)]
factor6 = data1.iloc[:,clump_factor_vars(factor_loadings,5)]

factors = [('factor1', factor1.columns), ('factor2', factor2.columns), ('factor3', factor3.columns), ('factor4', factor4.columns), ('factor5', factor5.columns), ('factor6', factor6.columns)]

for each in factors:
	print(each)

# Create a DF of factor loadings vs. original variable
loadings_df = pd.DataFrame(factor_loadings)
loadings_df.columns = ['Factor1', 'Factor2', 'Factor3', 'Factor4', 'Factor5', 'Factor6']
for each in range(len(factor_loadings)):
	loadings_df.rename(index={each: data1.iloc[0].index[each]}, inplace=True)

# Plot heatmaps
ax = sns.heatmap(loadings_df,annot=True,yticklabels=True)
plt.title('No Rotation')
# plt.title('Varimax Rotation')
# plt.show()

def calculate_bartlett_sphericity(x):
    """
    Test the hypothesis that the correlation matrix
    is equal to the identity matrix.identity
    H0: The matrix of population correlations is equal to I.
    H1: The matrix of population correlations is not equal to I.
    The formula for Bartlett's Sphericity test is:
    .. math:: -1 * (n - 1 - ((2p + 5) / 6)) * ln(det(R))
    Where R det(R) is the determinant of the correlation matrix,
    and p is the number of variables.
    Parameters
    ----------
    x : array-like
        The array from which to calculate sphericity.
    Returns
    -------
    statistic : float
        The chi-square value.
    p_value : float
        The associated p-value for the test.
    """
    n, p = x.shape
    x_corr = corr(x)

    corr_det = np.linalg.det(x_corr)
    statistic = -np.log(corr_det) * (n - 1 - (2 * p + 5) / 6)
    degrees_of_freedom = p * (p - 1) / 2
    p_value = stats.chi2.pdf(statistic, degrees_of_freedom)
    return statistic, p_value, degrees_of_freedom

chi_square1, p1, dof1 = calculate_bartlett_sphericity(data1)

observed = corr(data1)
expected = np.dot(factor_loadings,factor_loadings.T)
chi_square2, p2 = stats.chisquare(observed,expected)

# Perform CFA on K factor model
model_dict = {"F1": ['AbsPM25', 'NO', 'Noise_n', 'PCB118', 'PCB180', 'PM10Cu', 'PM25CU'],
               "F2": ['PM10Ni', 'PM10V', 'PM25Ni', 'PM25V'],
               "F3": ['BDE138', 'BDE17', 'BDE209', 'BDE66', 'BDE99', 'Green'],
               "F4": ['MEOHP', 'MnBP', 'X5cxMEPP'],
               "F5": ['CHCl3', 'DDE', 'THM'],
               "F6": ['BPA', 'Cotinine', 'MBzP', 'PFOA', 'Sb']}
model_spec = ModelSpecificationParser.parse_model_specification_from_dict(data1, model_dict)
cfa = ConfirmatoryFactorAnalyzer(model_spec, disp=False)
cfa.fit(data1.values)

# Problem 1d
pheno_data = pd.read_csv("https://donatello-telesca.squarespace.com/s/Phenotype.csv")
y1 = pheno_data.whistling_chest
y2 = pheno_data.flu
y3 = pheno_data.rhinitis
y4 = pheno_data.wheezing

factor_loadings_df = pd.DataFrame(factor_loadings)

c_score1 = (factor_loadings_df[0]**2).sum()/28
c_score2 = (factor_loadings_df[1]**2).sum()/28
c_score3 = (factor_loadings_df[2]**2).sum()/28
c_score4 = (factor_loadings_df[3]**2).sum()/28
c_score5 = (factor_loadings_df[4]**2).sum()/28
c_score6 = (factor_loadings_df[5]**2).sum()/28
