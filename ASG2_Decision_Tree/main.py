# Library Imports
from nis import cat
from re import X
import pandas as pd                 # Used for data frame
import plotly                       # Saves html plots
import plotly.express as px         # Used for displaying plots
import os                           # Allows file manipulation and console debugging for offline jupyter
import numpy as np
from scipy import stats             # Used for outliers
import matplotlib.pyplot as plt     # Used for pyplot heatmap plotting
import seaborn as sns               # Used for showing heatmap
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

import phik

from phik import resources
from phik.binning import bin_data
from phik.report import plot_correlation_matrix

# Module Imports

# Software configurations
pd.options.display.max_rows = 4000  # Allows better debugging analysis

# Global Variables
debug = True                        # Displays additional logging output
saveImages = False                  # Saves plot image files
targetColumnName = "TARGET"         # Name for column denoting dependant variable
outlierThreshold = 3                # Number of standard deviations from which data will be classified as an outlier
dropMissingValues = False
datasetName = './dataset/application_train.csv'
initialDataFileName = './outputFiles/initialData.txt'
missingValFileName = './outputFiles/missingValueSummary.txt'
noMissingValuesFileName = './outputFiles/noMissingValueSummary.txt'
initialDistributionFileName = './outputFiles/initialDistribution.txt'
lineString = "---------------------------------------------------------------------------------------------------------\n"


# If the debugging flag is on, creates directories to store output data
#
# Parameters:
# -----------
# @param debug: flag for displaying debugger output
#
# Returns:
# ---------
# None
#
def createOutputDirectories(debug = False):
	if debug == False:
		return
	if not os.path.exists("images"):
		os.mkdir("images")
	if not os.path.exists("images/initialPlots"):
		os.mkdir("images/initialPlots")
	if not os.path.exists("outputFiles"):
		os.mkdir("outputFiles")


# Reads csv file into data frame and sets independant and dependant variables
#
# Parameters:
# -----------
# @param fileName: string for full relative file path of csv file
# @param dependantVarColumnName: csv file column matching name of column for dependant variable
# @param debug: flag for displaying debugger output
#
# Returns:
# ---------
# data: dataframe object of csv file reading
# independantVars: independant variables (all data that isn't targetColumnName)
# dependantVar: dependant variable
#
def readData(fileName, dependantVarColumnName = targetColumnName, debug = False):
	independantVars = []
	dependantVar = []
	data = pd.read_csv(fileName)
	index = None
	for i ,col in enumerate(data.columns):
		if col == dependantVarColumnName:
			index = i
	if index != None: 
		dependantVar = data.iloc[:, index]
		independantVars = data.iloc[:]
		independantVars.pop(dependantVarColumnName)
	if debug:
		fd = open(initialDataFileName, "w+")
		fd.write("This file contains the initial data frame without cleaning:\n")
		fd.write(str(data))
		fd.close()
	return data, independantVars, dependantVar


# Drops rows from dataset which are missing. Prints missing value data for debugging
#
# Parameters:
# -----------
# @param data: dataframe to have missing values dropped and returned
# @param debug: flag for displaying debugger output
#
# Returns:
# ---------
# data: dataframe object with missing values dropped
#
def dropMissingValues(data, debug = False):
    # Drop missing values
    ret = data.dropna(axis=0)
    # Show number of missing values per independant variable
    if debug:
        fd = open(missingValFileName, "w+")
        fd.write("This data shows the independant variables which contained missing values and the count of each:\n")
        fd.write(str(data.isnull().sum()))
        fd.close()
        fd = open(noMissingValuesFileName, "w+")
        fd.write("This data shows the independant variables which are used for analysis with no mising values:\n")
        fd.write(str(ret.isnull().sum()))
        fd.close()
    return ret


def setMeanValues(data, debug = False):
    return data

# Writes distribution of data frame to text file
#
# Parameters:
# -----------
# @param data: dataframe to have distribution written to text file
# @param debug: flag for displaying debugger output
#
# Returns:
# ---------
# None
#
def writeDistribution(data, debug = False):
    if debug == False:
        return
    numpy_array = data.to_numpy()
    fd = open(initialDistributionFileName, "w+")
    fd.write(str(numpy_array))
    fd.close()


def doBar(data, column_name, figsize = (18,6), percentage_display = True, plot_defaulter = True, rotation = 0, horizontal_adjust = 0, fontsize_percent = 'xx-small'):

    print(f"Total Number of unique categories of {column_name} = {len(data[column_name].unique())}")
    
    plt.figure(figsize = figsize, tight_layout = False)
    sns.set(style = 'whitegrid', font_scale = 1.2)
    
    #plotting overall distribution of category
    plt.subplot(1,2,1)
    data_to_plot = data[column_name].value_counts().sort_values(ascending = False)
    ax = sns.barplot(x = data_to_plot.index, y = data_to_plot, palette = 'Set1')
    
    if percentage_display:
        total_datapoints = len(data[column_name].dropna())
        for p in ax.patches:
            ax.text(p.get_x() + horizontal_adjust, p.get_height() + 0.005 * total_datapoints, '{:1.02f}%'.format(p.get_height() * 100 / total_datapoints), fontsize = fontsize_percent)
        
    plt.xlabel(column_name, labelpad = 10)
    plt.title(f'Distribution of {column_name}', pad = 20)
    plt.xticks(rotation = rotation)
    plt.ylabel('Counts')
    
    #plotting distribution of category for Defaulters
    if plot_defaulter:
        percentage_defaulter_per_category = (data[column_name][data.TARGET == 1].value_counts() * 100 / data[column_name].value_counts()).dropna().sort_values(ascending = False)

        plt.subplot(1,2,2)
        sns.barplot(x = percentage_defaulter_per_category.index, y = percentage_defaulter_per_category, palette = 'Set2')
        plt.ylabel('Percentage of Defaulter per category')
        plt.xlabel(column_name, labelpad = 10)
        plt.xticks(rotation = rotation)
        plt.title(f'Percentage of Defaulters for each category of {column_name}', pad = 20)

    fileName = 'images/initialPlots/' + column_name + '.png'
    plt.savefig(fileName)


def plot_column(data,
                column_name,
                plots = [],
                figsize = (20,8),
                log_scale = False):

    if 'bar' in plots:
        doBar(data, column_name, figsize)
        return
    data_to_plot = data.copy()
    plt.figure(figsize = figsize)
    sns.set_style('whitegrid')
    
    for i, ele in enumerate(plots):
        plt.subplot(1, len(plots), i + 1)
        plt.subplots_adjust(wspace=0.25)
        if ele == 'CDF':
            #making the percentile DataFrame for both positive and negative Class Labels
            percentile_values_0 = data_to_plot[data_to_plot.TARGET == 0][[column_name]].dropna().sort_values(by = column_name)
            percentile_values_0['Percentile'] = [ele / (len(percentile_values_0)-1) for ele in range(len(percentile_values_0))]
            
            percentile_values_1 = data_to_plot[data_to_plot.TARGET == 1][[column_name]].dropna().sort_values(by = column_name)
            percentile_values_1['Percentile'] = [ele / (len(percentile_values_1)-1) for ele in range(len(percentile_values_1))]
            
            plt.plot(percentile_values_0[column_name], percentile_values_0['Percentile'], color = 'red', label = 'Non-Defaulters')
            plt.plot(percentile_values_1[column_name], percentile_values_1['Percentile'], color = 'black', label = 'Defaulters')
            plt.xlabel(column_name)
            plt.ylabel('Probability')
            plt.title('CDF of {}'.format(column_name))
            plt.legend(fontsize = 'medium')
            if log_scale:
                plt.xscale('log')
                plt.xlabel(column_name + ' - (log-scale)')
        elif ele == 'distplot':
            sns.distplot(data_to_plot[column_name][data['TARGET'] == 0].dropna(),
                         label='Non-Defaulters', hist = False, color='red')
            sns.distplot(data_to_plot[column_name][data['TARGET'] == 1].dropna(),
                         label='Defaulters', hist = False, color='black')
            plt.xlabel(column_name)
            plt.ylabel('Probability Density')
            plt.legend(fontsize='medium')
            plt.title("Dist-Plot of {}".format(column_name))
            if log_scale:
                plt.xscale('log')
                plt.xlabel(f'{column_name} (log scale)')
        elif ele == 'violin':  
            sns.violinplot(x='TARGET', y=column_name, data=data_to_plot)
            plt.title("Violin-Plot of {}".format(column_name))
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)')
        elif ele == 'box':  
            sns.boxplot(x='TARGET', y=column_name, data=data_to_plot)
            plt.title("Box-Plot of {}".format(column_name))
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)')

    fileName = 'images/initialPlots/' + column_name + '.png'
    plt.savefig(fileName)


def showPlots(data, outlierAnalysisColumns = [], debug = False):
    if debug == False:
        return
    if len(outlierAnalysisColumns) == 0:
        outlierAnalysisColumns = data.columns
    index = 0
    # Plots 10 items per chart to prevent things from getting too large
    for i in range(0,len(outlierAnalysisColumns)):
        index = index + 1
        fig = px.scatter_matrix(data,
                                dimensions=outlierAnalysisColumns[i:i+1],
                                labels={col:col.replace('_', ' ') for col in data.columns},
                                height = 1500,
                                width = 1500,
                                color=targetColumnName,
                                title="Outlier Analysis Scatterplot " + data.columns[i],
                                color_continuous_scale=px.colors.diverging.Tealrose)
        fileName = "images/initialPlots/scatterplot_" + data.columns[i]
        fig.write_image(fileName + ".jpg")


def showTargetPlot(data, debug = False):
    class_dist = data[targetColumnName].value_counts()

    if debug == True:
        print(class_dist)

    plt.figure(figsize=(12,3))
    plt.title('Distribution of TARGET variable')
    plt.barh(class_dist.index, class_dist.values)
    plt.yticks([0, 1])

    for i, value in enumerate(class_dist.values):
        plt.text(value-2000, i, str(value), fontsize=12, color='white',
                 horizontalalignment='right', verticalalignment='center')
    plt.show()


def showHeatmap(data):
    corrmat = data.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    #plot heat map
    g=sns.heatmap(data[top_corr_features].corr(),cmap="RdYlGn")
    plt.show()


def allocateTypes(data, debug = False):
    strTypes = data.select_dtypes(include='object')
    continuousTypes = data.select_dtypes(include = 'float64')
    categoricalTypes = data.select_dtypes(include = 'int64')
    strTypes.insert(0, targetColumnName, data[targetColumnName])
    continuousTypes.insert(0, targetColumnName, data[targetColumnName])

    if debug == True: 
        print("String-type variables:\n")
        print(*strTypes.columns, sep = "\n")
        print(lineString)
        print("Continuous-type variables:\n")
        print(*continuousTypes.columns, sep = "\n")
        print(lineString)
        print("Categorical-type variables:\n")
        print(*categoricalTypes.columns, sep = "\n")
        print(lineString)
    return strTypes, continuousTypes, categoricalTypes


# Main Method:
#-------------
# Reads in the data files, plots certain values and creates useful analytical plots and does
# some light data cleaning
#
#
# Parameters:
# -----------
# @param debug:             flag for displaying debugger output
# @param dropMissingValues: true to drop rows with empty values, false to set null values to mean
# @param: savePlots:        true to plot various initial data points
# @param outlierThreshold:  z-value with which to threshold outliers
#
# Returns:
# ---------
# None
#
def main(debug = True, dropMissingValues = False, savePlots = False, outlierThreshold = 3):
    # Create output directories for files and plots to be saved to
    createOutputDirectories(debug)

    # Read the data, assigning independant and dependant variables: x and y respectively
    data, x, y = readData(datasetName, targetColumnName, debug)

    # Shows distribution of independant variable, and shows heatmap, if desired
    if savePlots == True:
        showHeatmap(data)
        showTargetPlot(data, debug)

    # Drop missing values or fill in empty values with mean
    if dropMissingValues == True: 
        data = dropMissingValues(data, debug)
    else:
        data = setMeanValues(data, debug)

    # Show data distribution and allow for manual analysis of outliers
    writeDistribution(data, debug)
    
    # Remove outliers past threshold of 3
    data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]

    # Get sub-data frames that contain variables from each respective data type
    data_strings, data_continuous, data_categorical = allocateTypes(data, debug)

    # Feature engineering for a credit to income ratio
    data_continuous.insert(0, 'loan_ratio', data_continuous['AMT_CREDIT'] /data_continuous['AMT_INCOME_TOTAL'] / 100)
    plot_column(data_continuous, 'loan_ratio', ['box'])

    # Show plots from data for outlier analysis
    # [] denotes all variables to look at
    if savePlots == True: 
        showPlots(data, [], debug)
        for i in data_strings.columns:
            plot_column(data_strings, i, ['bar'])
        for i in data_continuous.columns:
            plot_column(data_continuous, i, ['box'])
        for i in data_categorical.columns:
            plot_column(data_categorical, i, ['bar'])


if __name__ == "__main__":
    main(debug=debug, dropMissingValues=dropMissingValues, savePlots = saveImages, outlierThreshold = 3)