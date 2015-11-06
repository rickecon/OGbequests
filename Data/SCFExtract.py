import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.optimize import curve_fit
import sys

'''
FROM 2013 SURVEY OF CONSUMER FINANCES (the period before was 2010)
Variables used:

X5804(#1)       What was its approximate value (of bequests) at the time it was received?
X5809(#2)       (second instance of bequest reception)
X5814(#3)       (third instance of bequest reception)

X5805(#1)       In what year was it received? (years for instances #1-#3 respectively)
X5810(#2)            we check if it was in the last 3 years (2011-2013)
X5815(#3)		     if not received = 0.
				

X8022           FOR THE RESPONDENT, THIS VARIABLE CONTAINS THE DATE-OF-BIRTH (age)

Net Worth		Already adjusted for inflation, respondent's perception of own
				net worth, summary variable

wgt             summary variable, weights provided by the Federal Reserve 
                to adjust for selection bias in the survey.

Income level/ Ability Type categories:

1.  Under $15000			
2.  15,000-24,999			
3.  25,000-49,999		
4.  50,000-74,999			
5.  75,000-99,999	
6.  100,000-249,999	
7.  250,000 or more		

Age categories:

(min age in SCF generally is 18)

ages 18-96	

(max in 2013 was 95 yrs old)'''

'''Loading in data sets into different panda dataframes'''

filenames = ['p13i6','p10i6','p07i6','p04i6','p01i6','p98i6',
'p95i6','p92i4','p89i6']

summaryfiles = ['rscfp2013','rscfp2010','rscfp2007','rscfp2004','rscfp2001',
'rscfp1998','rscfp1995','rscfp1992','rscfp1989']

year_start = int(sys.argv[1])
year_end = int(sys.argv[2])
graphs = bool(sys.argv[3])

# year_start = 1998
# year_end = 2013
# graphs = False

'''Making sure feasible inputs are given'''
if (graphs != True) & (graphs != False):
    raise ValueError('Please indicate whether you want graphs or not (\'True\' or \'False\')')
if (year_start > 2013) or (year_start<1989) or (year_start % 3 != 0) :
    raise ValueError("SCF data non-existent for start year provided")
if (year_end > 2013) or (year_end<1989) or (year_end % 3 != 0):
    raise ValueError("SCF data non-existent for end year provided")

'''initializing the year range to be used in SCF data extraction'''
year_range = np.arange(year_end, year_start-3, -1)
year_list = []
for year in year_range:
    if year % 3 == 0:
        year_list.append(year)
    else:
        pass

'''intializing lists with SCF and SCF summary variable filenames''' 
df_list = []
for i in xrange(len(year_list)):
    df_list.append(pd.read_stata(filenames[i]+'.dta'))
dfs_list = []
for i in xrange(len(year_list)):
    dfs_list.append(pd.read_stata(summaryfiles[i]+'.dta'))

'''initialize variable for all years combined graph'''
yearall = str(year_start) +'-'+str(year_end)

'''set different income levels for the different ability types'''
income_levels = [-999999999, 15000,25000,50000,75000,100000,250000, 999999999]


'''setting different inflation levels for years 1987-2013 drawn from CPI for 
all urban consumers: all items from https://research.stlouisfed.org/fred2/search?st=CPI'''
inflation_levels = np.array([1.51229385,1.492299173,
1.467973318,1.439144582,1.415496948,1.397721517,1.379834479,1.363737434,1.345889029,
1.326679888,1.310939123,1.300280732,1.284934882,1.260857994,1.240039148,1.227912707,1.210171616,
1.189103802,1.161807505,1.134803101,1.109966432,1.076012397,1.078969961,1.063898833,1.034477726,
1.014431538,1])
inflation_levels = inflation_levels[::-1]
inflation_levels = inflation_levels[:len(year_range)]

'''variable for the y axis income groups for graph'''
income_y = np.arange(1,8)

'''the minimum age in the survey'''
min_age = 18

'''the maximum age that we want to represent in the data'''
max_age = 95

'''creating an array full of the different ages'''
age_groups = np.arange(min_age, max_age+1)
income_groups = 7


def initSCFvars(scf, scfSummary, year, year_range, inflation_levels):
	'''
    Generates variables from the SCF that will eventually be used
    to calculate total bequests from all respondents, total 
    bequests for age-income categories, and a proportion matrix.

    Inputs:
        scf   	 			  = dictionary with all SCF data, indexed with codebook
        		   			    variables.
        scfSummary            = dictionary with all SCF summary variables adjusted for inflation.

        year     			  = scalar, what year the SCF survey was taken

        year_range            = list, years to be adjusted for inflation

        inflation_levels      = list, inflation corresponding to the years in year_range
       
    Functions called: None

    Objects in function:
        age                  = panda dataframe, contains the ages of the respondents

        age_income_matrix     = [X = # of income groups, Y = # of age groups], 
        						Nested lists, this variable contains a list of Y lists
        						corresponding to each age group, and each of the Y
        						lists contains X amounts of income groups. These nested lists
        						will be filled with pandas and will be used to extract inheritance
        						data corresponding to the age and income conditions.

        income           	  = panda dataframe, contains total household income for each respondent

       	year_received         = panda dataframe, contains the year that each respondent received
       							their wealth transfer.

        age_income 			  = panda dataframe, combines the age, income, value_of_transfer,
        						and year_received pandas into one panda.

        last_three_years 	  = panda dataframe, contains booleans corresponding to the respondents 
        						that received their transfer in the last three years

        new_list              = list, contains dataframes of the bequest amounts adjusted for inflation.

        recent_recipients 	  = panda dataframe, contains the indeces of the respondents that
        						received their transfer in the last three years.

        sum_bequest_list 	  = panda dataframe, contains the respondents that received their 
        						wealth transfers in the last three years.

        sum_bequest_list_final= dataframe, contains the bequests amounts adjusted for inalation of 
                                respondents that have received in last three years.

        total_bequest_matrix  = [A = # of age groups, B = # of income_groups] array that will
                                be filled with the total bequests of each age-income category

        total_bequests_received = scalar, total wealth transfers received in the last 3 
        						years for this data set

        three_years_received_combined_summed     = panda dataframe, contains the total value of wealth transfers for each
                                                   respondent.

        weights               = dataframe, contains the weights corresponding to each respondent,
                                these correct for selection bias in survey.


    Returns: age_income, total_bequests_received, age_income_matrix, total_bequest_matrix
    '''
        age_income_matrix = [[[] for z in xrange(income_groups)] for x in xrange(len(age_groups))]
        total_bequest_matrix = np.zeros((len(age_groups), income_groups))
        last_three_years = (year-scf['X5805'])<3
        recent_recipients = last_three_years.index[last_three_years == True]
        sum_bequest_list = scf['X5804'][recent_recipients]
        sum_bequest_list_w_years = pd.concat([sum_bequest_list, scf['X5805'][recent_recipients]], axis = 1)
        new_list = [sum_bequest_list_w_years[sum_bequest_list_w_years['X5805'] == year_range[i]] * inflation_levels[i] for i in xrange(len(inflation_levels))]
        new_pd = pd.concat([new_list[i] for i in xrange(len(new_list))])
        new_pd_only_bequests = new_pd['X5804']
        sum_bequest_list_final = new_pd_only_bequests
        last_three_years2 = (year-scf['X5810'])<3
        recent_recipients2 = last_three_years2.index[last_three_years2 == True]
        sum_bequest_list2 = scf['X5809'][recent_recipients2]
        sum_bequest_list_w_years2 = pd.concat([sum_bequest_list2, scf['X5810'][recent_recipients2]], axis = 1)
        new_list2 = [sum_bequest_list_w_years2[sum_bequest_list_w_years2['X5810'] == year_range[i]] * inflation_levels[i] for i in xrange(len(inflation_levels))]
        new_pd2 = pd.concat([new_list2[i] for i in xrange(len(new_list2))])
        new_pd_only_bequests2 = new_pd2['X5809']
        sum_bequest_list2_final = new_pd_only_bequests2
        last_three_years3 = (year - scf['X5815'])<3
        recent_recipients3 = last_three_years3.index[last_three_years3 == True]
        sum_bequest_list3 = scf['X5814'][recent_recipients3]
        sum_bequest_list_w_years3 = pd.concat([sum_bequest_list3, scf['X5815'][recent_recipients3]], axis = 1)
        new_list3 = [sum_bequest_list_w_years3[sum_bequest_list_w_years3['X5815'] == year_range[i]] * inflation_levels[i] for i in xrange(len(inflation_levels))]
        new_pd3 = pd.concat([new_list3[i] for i in xrange(len(new_list3))])
        new_pd_only_bequests3 = new_pd3['X5814']
        sum_bequest_list3_final = new_pd_only_bequests3
        three_years_received_combined = pd.concat([sum_bequest_list_final, sum_bequest_list2_final, sum_bequest_list3_final], axis =1)
        three_years_received_combined_summed = three_years_received_combined.sum(axis =1)
        indeces = three_years_received_combined_summed.index
        age = scf['X8022'][indeces]
        weights = scfSummary['wgt'][indeces]
        income =scfSummary['networth'][indeces]
        three_years_received_combined_summed = three_years_received_combined_summed * (weights/5.)
        age_income = pd.concat([three_years_received_combined_summed, age, income], axis =1)
        age_income.columns = ['X5804' , 'X8022' , 'X5729']
        total_bequests_received = three_years_received_combined_summed.sum()
        return (age_income, total_bequests_received, age_income_matrix, total_bequest_matrix)


def genBequestMatrix(age_income_matrix, total_bequest_matrix, age_groups, 
	income_groups, min_age, income_levels, age_income, year):
	'''
    Generates a matrix for the total bequests of each age-income category, 
    after generating a Nested list containing the pandas corresponding to the age, 
    income and recent wealth transfer conditions.

    Inputs:
        age_income_matrix     = [X = # of income groups, Y = # of age groups], 
        						Nested lists, this variable contains a list of Y lists
        						corresponding to each age group, and each of the Y
        						lists contains X amounts of income groups. These nested lists
        						will be filled with pandas and will be used to extract inheritance
        						data corresponding to the age and income conditions.

        total_bequest_matrix  = [A = # of age groups, B = # of income_groups] array that will
        						be filled with the total bequests of each age-income category

        age_groups            = [A, ], vector of integers corresponding to different ages
        						of respondents.

        income_groups         = scalar, total number of income groups/ ability types.

        min_age               = scalar, minimum age of respondents

        income_levels         = [B, ], vector of scalars, corresponding to different
        						income cut-offs.

        age_income            = panda dataframe, combines the age, total_income, value_of_transfer,
        						and year_received pandas into one panda.

        year                  = scalar, what year the SCF survey was taken

       
    Functions called: None

    Objects in function:
        age_income_matrix     = [X = # of income groups, Y = # of age groups], 
        						Nested lists, this variable contains a list of Y lists
        						corresponding to each age group, and each of the Y
        						lists contains X amounts of income groups. Filled with pandas 
        						corresponding to the age, income and recent wealth transfer conditions.

        total_bequest_matrix  = [A = # of age groups, B = # of income_groups], array filled with 
        						the total bequests of each age-income category

    Returns: age_income_matrix, total_bequest_matrix
    '''
	for i in age_groups:
		for j in xrange(income_groups):
			age_income_matrix[i-min_age][j] = age_income[(age_income['X8022']==i) & (age_income['X5729'] < income_levels[j+1]) & (age_income['X5729'] >= income_levels[j])]
			total_bequest_matrix[(i-min_age),j] = age_income_matrix[i-min_age][j]['X5804'].sum()
	return age_income_matrix, total_bequest_matrix

def genProportionMat(age_groups, income_groups, total_bequest_matrix, total_bequests_received):
	'''
    Generates a matrix [A, B], containing the proportion (0 < x < 1) of the total bequests 
    that each age-income category receives.

    Inputs:

        age_groups            = [A, ], vector of integers corresponding to different ages
        						of respondents.

        income_groups         = scalar, total number of income groups/ ability types.

		total_bequest_matrix  = [A = # of age groups, B = # of income_groups] array that will
        						be filled with the total bequests of each age-income category

        total_bequests_received = scalar, total wealth transfers received in the last 3 
        						years for this data set
     
    Functions called: None

    Objects in function:
        proportion_matrix     = [A, B], array containing the proportion (0 < x < 1) of the total bequests 
    							that each age-income category receives.

    Returns: proportion_matrix
    '''
	proportion_matrix = np.zeros((len(age_groups), income_groups))
	for i in xrange(len(age_groups)):
		for j in xrange(income_groups):
			proportion_matrix[i, j] = total_bequest_matrix[i,j]/total_bequests_received
	return proportion_matrix

'''graphing functions for combined age-income distribution and just age and income serparately'''

def age_income_plot(income_y, age_groups, proportion_matrix, year):
	X,Y=np.meshgrid(income_y, age_groups)
	fig=plt.figure()
	ax=fig.gca(projection='3d')
	ax.plot_surface(X,Y, proportion_matrix, rstride=5)
	ax.set_xlabel("Ability Types")
	ax.set_ylabel("Age")
	ax.set_zlabel("Received percentage of total bequests {}".format(str(year)))
	plt.show()

def age_plot(total_bequest_matrix_filled, age_groups, year):
	just_age = total_bequest_matrix_filled.sum(axis=1)/total_bequest_matrix_filled.sum()
	fig, ax  = plt.subplots()
	plt.plot(age_groups, just_age, label =None)
	legend=ax.legend(loc= "upper right", shadow =True, title='ability types')
	plt.xlabel('age')
	plt.ylabel('percentage share of total received inheritances; 1=%100')
	plt.title('Received percentage share of total bequests with age for year {}'.format(str(year)))
	plt.show()


def income_plot(total_bequest_matrix_filled, income_y, year):
	just_income = total_bequest_matrix_filled.sum(axis =0)/total_bequest_matrix_filled.sum()
	fig, ax  = plt.subplots()
	plt.plot(income_y, just_income, label = None)
	legend=ax.legend(loc= "upper right", shadow =True, title='ability types')
	plt.xlabel('net worth/ability types')
	plt.ylabel('percentage share of total received inheritances; 1=%100')
	plt.title('Received share of total bequests based on net worth for year {}'.format(str(year)))
	plt.show()


'''initialize lists to contain the matrices produced for the different years'''
proportion_matrix_list = []
total_bequests_received_list = []
total_bequest_matrix_list = []
counter = 0

'''creating the different matrices for the different years and inserting them into their respective lists'''
for year in (year_list):
    age_income, total_bequests_received, age_income_matrix, total_bequest_matrix = initSCFvars(df_list[counter], dfs_list[counter], year, year_range, inflation_levels)
    age_income_matrix_filled, total_bequest_matrix_filled = genBequestMatrix(age_income_matrix, total_bequest_matrix, \
		age_groups, income_groups, min_age, income_levels, age_income, year)
    proportion_matrix = genProportionMat(age_groups, income_groups, total_bequest_matrix_filled, total_bequests_received)
    proportion_matrix_list.append(proportion_matrix)
    total_bequests_received_list.append(total_bequests_received)
    total_bequest_matrix_list.append(total_bequest_matrix_filled)
    counter += 1

'''creating the matrix of all years combined'''
all_summed_total_bequests = np.sum(total_bequests_received_list)
all_summed_total_bequests_mat = np.zeros((len(age_groups), income_groups))
for i in xrange(len(year_list)):
    all_summed_total_bequests_mat += total_bequest_matrix_list[i]
all_years_proportion = genProportionMat(age_groups, income_groups, all_summed_total_bequests_mat, all_summed_total_bequests)

'''graphing the results'''
if graphs==True:
    counter2 = 0
    for year in year_list:
        age_plot(total_bequest_matrix_list[counter2], age_groups, year)
        income_plot(total_bequest_matrix_list[counter2], income_y, year)
        age_income_plot(income_y, age_groups, proportion_matrix_list[counter2], year)
        counter2 +=1
    income_plot(all_summed_total_bequests_mat, income_y, yearall)
    age_plot(all_summed_total_bequests_mat, age_groups, yearall)
    age_income_plot(income_y, age_groups, all_years_proportion, yearall)

    
