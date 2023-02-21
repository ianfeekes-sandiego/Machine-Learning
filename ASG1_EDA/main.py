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

# Aggregate of ext source 1, 2, 3: If I can only choose a single then ext source 1

# @TODO: figure out if we want these later or not
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
    #np.savetxt("test_file.txt", numpy_array, fmt = "%d")
    fd = open(initialDistributionFileName, "w+")
    #fd.write(str(data.describe()))
    fd.write(str(numpy_array))
    fd.close()


    '''
    Function to plot Categorical Variables Bar Plots
    
    Inputs:
        data: DataFrame
            The DataFrame from which to plot
        column_name: str
            Column's name whose distribution is to be plotted
        figsize: tuple, default = (18,6)
            Size of the figure to be plotted
        percentage_display: bool, default = True
            Whether to display the percentages on top of Bars in Bar-Plot
        plot_defaulter: bool
            Whether to plot the Bar Plots for Defaulters or not
        rotation: int, default = 0
            Degree of rotation for x-tick labels
        horizontal_adjust: int, default = 0
            Horizontal adjustment parameter for percentages displayed on the top of Bars of Bar-Plot
        fontsize_percent: str, default = 'xx-small'
            Fontsize for percentage Display
        
    '''
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


'''
Function to plot continuous variables distribution

Inputs:
    data: DataFrame
        The DataFrame from which to plot.
    column_name: str
        Column's name whose distribution is to be plotted.
    plots: list, default = ['distplot', 'CDF', box', 'violin']
        List of plots to plot for Continuous Variable.
    scale_limits: tuple (left, right), default = None
        To control the limits of values to be plotted in case of outliers.
    figsize: tuple, default = (20,8)
        Size of the figure to be plotted.
    histogram: bool, default = True
        Whether to plot histogram along with distplot or not.
    log_scale: bool, default = False
        Whether to use log-scale for variables with outlying points.
'''
def plot_column(data,
                column_name,
                plots = ['distplot', 'CDF', 'box', 'violin'],
                figsize = (20,8),
                histogram = True,
                log_scale = False):

    if 'bar' in plots:
        doBar(data, column_name, figsize)
        return

    data_to_plot = data.copy()

    number_of_subplots = len(plots)
    plt.figure(figsize = figsize)
    sns.set_style('whitegrid')
    
    for i, ele in enumerate(plots):
        plt.subplot(1, number_of_subplots, i + 1)
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
        elif ele == 'bar':
            continue
            #data_to_plot = data_to_plot.value_counts().sort_values(ascending = False)
            #ax = sns.barplot(x = data_to_plot.index, y = data_to_plot, palette = 'Set1')
            #plt.xlabel(column_name, labelpad = 10)
            #plt.title(f'Distribution of {column_name}', pad = 20)
            #plt.ylabel('Counts')

            #percentage_defaulter_per_category = (data[column_name][data.TARGET == 1].value_counts() * 100 / data[column_name].value_counts()).dropna().sort_values(ascending = False)
            #plt.subplot(1,2,2)
            #sns.barplot(x = percentage_defaulter_per_category.index, y = percentage_defaulter_per_category, palette = 'Set2')
            #plt.ylabel('Percentage of Defaulter per category')
            #plt.xlabel(column_name, labelpad = 10)
            #plt.title(f'Percentage of Defaulters for each category of {column_name}', pad = 20)

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

    # Shows distribution of independant variable
    if savePlots == True:
        showTargetPlot(data, debug)

    # Drop missing values or fill in empty values with mean
    #@TODO: uncomment
    #if dropMissingValues == True: 
    #    data = dropMissingValues(data, debug)
    #else:
    #     data = setMeanValues(data, debug)

    # Show data distribution and allow for manual analysis of outliers
    writeDistribution(data, debug)
    
    # Show plots from data for outlier analysis
    # [] denotes all variables to look at
    if savePlots == True: 
        showPlots(data, [], debug)

    # Remove outliers past threshold of 3
    #data[(np.abs(stats.zscore(data)) < 3).all(axis=1)]

    # Show heatmap if desired
    #sns.heatmap(data.corr(), cmap="YlGnBu")
    #plt.show()

    #X=data
    #vif = pd.DataFrame()
    #vif["features"] = X.columns
    #vif["vif_Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    #print(vif)

    #model = ExtraTreesClassifier()
    #model.fit(x,y)
    #print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    ##plot graph of feature importances for better visualization
    #feat_importances = pd.Series(model.feature_importances_, index=x.columns)
    #feat_importances.nlargest(10).plot(kind='barh')
    #plt.show()

    if False:
        showHeatmap(data)

    # Get sub-data frames that contain variables from each respective data type
    data_strings, data_continuous, data_categorical = allocateTypes(data, debug)

    data_continuous.insert(0, 'loan_ratio', data_continuous['AMT_CREDIT'] /data_continuous['AMT_INCOME_TOTAL'] / 100)
    plot_column(data_continuous, 'loan_ratio', ['box'])

    #@TODO: uncomment these
    #for i in data_strings.columns:
    #    plot_column(data_strings, i, ['bar'])
    #for i in data_continuous.columns:
    #    plot_column(data_continuous, i, ['box'])
    #for i in data_categorical.columns:
    #    plot_column(data_categorical, i, ['bar'])


    #data = data_continuous.phik_matrix()
    #plot_correlation_matrix(data.values, x_labels=data.columns, y_labels=data.index, 
    #                    vmin=0, vmax=1, color_map='Blues', title=r'correlation $\phi_K$', fontsize_factor=1.5,
    #                    figsize=(7,5.5))


    #x = data.iloc[:,2:]  #independent columns
    ##y = data.iloc[:,1]    #target column i.e price range
    #y = data[[targetColumnName]]
    #regr = linear_model.LinearRegression()
    #regr.fit(x, y)
    #print(regr)

    #print(y)

    #x[x.columns[n]] = x[x.columns[n]].apply(pd.to_numeric, errors='coerce').fillna(0).astype(float).dropna()
    #print(x)
    #print()
    #print("------------------------------------")
    #print(y)

    #bestfeatures = SelectKBest(score_func=chi2, k=10)
    #fit = bestfeatures.fit(x,y)
    #dfscores = pd.DataFrame(fit.scores_)
    #dfcolumns = pd.DataFrame(x.columns)
    ##concat two dataframes for better visualization 
    #featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    #featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    #print(featureScores.nlargest(10,'Score'))  #print 10 best features
    
    return


    # Scatterplot matrix. Only performed if desired to save on computation time
    if savePlots:
        outlierAnalysisColumns = ["CNT_CHILDREN", "AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_CREDIT", "AMT_GOODS_PRICE", "REGION_POPULATION_RELATIVE", "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "OWN_CAR_AGE"]
        fig = px.scatter_matrix(data,
                                dimensions=outlierAnalysisColumns,
                                labels={col:col.replace('_', ' ') for col in data.columns},
                                height=900,
                                color=targetColumnName,
                                title="Outlier Analysis Scatterplot",
                                color_continuous_scale=px.colors.diverging.Tealrose)
        if not os.path.exists("images"):
            os.mkdir("images")
        else: 
            fig.write_image("images/scatterplt.jpg")
            fig.write_image("images/scatterplt.png")
            plotly.offline.plot(fig, filename='images/scatterplt.html')
        # Debugging flat allows for plot to be shown locally in console as well
        if debug:
            fig.show()

    """
    Outlier Analysis: 

    Threshold amtincometotal at >2.5mil
    Threshold amtannuity at 140k maybe
    Threshold amtcredit at 3.3mil
    Threshold amt goods price at 3mil
    """
    
    """for i, col in enumerate(data.columns):
        data = data[data[col] <= data[col].mean() + data[col].std()*outlierThreshold]
    """
    #data = data[data["AMT_INCOM_TOTAL"] <= 2500000]
    #data = data[data["AMT_ANNUITY"] <= 140000]
    #data = data[data["AMT_CREDIT"] <= 3300000]
    #data = data[data["AMT_GOODS_PRICE"] <= 3000000]
    #data = data[data["REGION_POPULATION_RELATIVE"] <= .004]


    #Percentage of non-default cases
    #print(data[data[targetColumnName] == 0].sum())
    #data_0 = data[data[targetColumnName] == 0].count() / data[targetColumnName].count()
    #print(data_0)

    data_0 = data[data.TARGET == 0].TARGET.count() / data.TARGET.count()
    # If we are dealing with an imbalanced classification problem then signal it as such
    if data_0 > .65:
        print(data_0)


    #Box plot
    #data["percentage"] = data[data["AMT_CREDIT"]] / data[data["AMT_INCOME_TOTAL"]] * 100

    fig = px.box(data, x="TARGET", y="AMT_INCOME_TOTAL", color="TARGET",
    color_discrete_sequence=px.colors.qualitative.Dark24,
    labels={col:col.replace('_', ' ') for col in data.columns},
    category_orders={"TARGET":["0", "1"]})
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom",
    y=1.02, xanchor="right", x=1))
    fig.show()

    #print(data.iloc[:,0:3])
    
    #"""X = data.iloc[:0,
    #
    #data = pd.read_csv("D://Blogs//train.csv")
    #X = data.iloc[:,0:20]  #independent columns
    #y = data.iloc[:,-1]    #target column i.e price range
    ##apply SelectKBest class to extract top 10 best features
    #bestfeatures = SelectKBest(score_func=chi2, k=10)
    #fit = bestfeatures.fit(X,y)
    #dfscores = pd.DataFrame(fit.scores_)
    #dfcolumns = pd.DataFrame(X.columns)
    ##concat two dataframes for better visualization 
    #featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    #featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    #print(featureScores.nlargest(10,'Score'))  #print 10 best features
    #"""

if __name__ == "__main__":
    main(debug=debug, dropMissingValues=dropMissingValues, savePlots = saveImages, outlierThreshold = 3)