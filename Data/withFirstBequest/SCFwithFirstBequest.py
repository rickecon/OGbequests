import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


'''
FROM 2013 SURVEY OF CONSUMER FINANCES (the period before was 2010)

X5729           What would be the correct total?
                
                How much was the total income you (and your family living here)
                received in 2009 from all sources, before taxes and other
                deductions were made?
                
                IF R SAYS TOTAL IS ZERO, ASK WHETHER THERE WAS A LOSS
                OR IF THE AMOUNT IS ACTUALLY ZERO.

                ANNUAL $ AMOUNT IN 2009:
                    -1.     Nothing
                    -9.     Negative (public data set only)
                *********************************************************
                    ORIGINALLY ALLOWED VALUES: [-99999999,...,999999999]
                *********************************************************
                    FOR THE PUBLIC DATA SET, NEGATIVE VALUES ARE SET TO -9
                *********************************************************

X5804(#1)       What was its approximate value at the time it was received?
X5809(#2)       
X5814(#3) 

X5805(#1)       In what year was it received?
X5810(#2)       ----we check if it was in the last 3 years (2010-2013)
X5815(#3)		if not received = 0.
				

X8022      FOR THE RESPONDENT, THIS VARIABLE CONTAINS THE DATE-OF-BIRTH (age)

Income level/ Ability Type categories:

1.  Under $15000			
2.  15000-24,999			
3.  25000-49,999		
4.  50,000-74,999			
5.  75000-79,999	
6.  100000 or more	
7.  250000 or more		

Age categories:

(min age in SCF generally is 18)

ages 18-96	

(max in 2013 was 95 yrs old)'''

'''Loading in data sets into different panda dataframes'''

scf2013 = pd.read_stata('p13i6.dta')
scf2010 = pd.read_stata('p10i6.dta')
scf2007 = pd.read_stata('p07i6.dta')
scf2004 = pd.read_stata('p04i6.dta')
scf2001 = pd.read_stata('p01i6.dta')

'''set the different year variables corresponding to the survey year'''

year2013 = 2013
year2010 = 2010
year2007 = 2007
year2004 = 2004
year2001 = 2001
year2001_2013 = "2001-2013"

'''set different income levels for the different ability types'''
income_levels = [0, 15000,25000,50000,75000,100000,250000, 999999999]

'''variable for the y axis income groups for graph'''
income_y = np.arange(1,8)

'''the minimum age in the survey'''
min_age = 18

'''the maximum age that we want to represent in the data'''
max_age = 95

'''creating an array full of the different ages'''
age_groups = np.arange(min_age, max_age+1)
income_groups = 7


def initSCFvars(scf, year):
	'''
    Generates variables from the SCF that will eventually be used
    to calculate total bequests from all respondents, total 
    bequests for age-income categories, and a proportion matrix.

    Inputs:
        scf   	 			  = dictionary with all SCF data, indexed with codebook
        		   			    variables.
        year     			  = scalar, what year the SCF survey was taken
       
    Functions called: None

    Objects in function:
        age_income_matrix     = [X = # of income groups, Y = # of age groups], 
        						Nested lists, this variable contains a list of Y lists
        						corresponding to each age group, and each of the Y
        						lists contains X amounts of income groups. These nested lists
        						will be filled with pandas and will be used to extract inheritance
        						data corresponding to the age and income conditions.

        total_bequest_matrix  = [A = # of age groups, B = # of income_groups] array that will
        						be filled with the total bequests of each age-income category

        total_income     	  = panda dataframe, contains total household income for each respondent

        value_of_transfer     = panda dataframe, contains the total value of wealth transfers for each
        						respondent.

       	year_received         = panda dataframe, contains the year that each respondent received
       							their wealth transfer.

        age   				  = panda dataframe, contains the ages of the respondents

        age_income 			  = panda dataframe, combines the age, total_income, value_of_transfer,
        						and year_received pandas into one panda.

        last_three_years 	  = panda dataframe, contains booleans corresponding to the respondents 
        						that received their transfer in the last three years
        recent_recipients 	  = panda dataframe, contains the indeces of the respondents that
        						received their transfer in the last three years.
        sum_bequest_list 	  = panda dataframe, contains the respondents that received their 
        						wealth transfers in the last three years.
        total_bequests_received = scalar, total wealth transfers received in the last 3 
        						years for this data set

    Returns: age_income, total_bequests_received, age_income_matrix, total_bequest_matrix
    '''

	age_income_matrix = [[[] for z in xrange(income_groups)] for x in xrange(len(age_groups))]
	total_bequest_matrix = np.zeros((len(age_groups), income_groups))
	total_income = scf['X5729']
	value_of_transfer = scf['X5804']
	year_received = scf['X5805']
	age = scf['X8022']
	age_income = pd.concat([age, total_income, year_received, value_of_transfer], axis =1)
	last_three_years = (year-scf['X5805'])<4
	recent_recipients = last_three_years.index[last_three_years == True]
	sum_bequest_list = scf['X5804'][recent_recipients]
	total_bequests_received = sum_bequest_list.sum()
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
			age_income_matrix[i-min_age][j] = age_income[(age_income['X8022']==i) & (age_income['X5729'] < income_levels[j+1]) & (age_income['X5729'] >= income_levels[j]) & (year-age_income['X5805']<4)]
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
	plt.ylabel('percentage that received inheritances; 1=%100')
	plt.title('recipiency rate of inheritances with age for year {}'.format(str(year)))
	plt.show()


def income_plot(total_bequest_matrix_filled, income_y, year):
	just_income = total_bequest_matrix_filled.sum(axis =0)/total_bequest_matrix_filled.sum()
	fig, ax  = plt.subplots()
	plt.plot(income_y, just_income, label = None)
	legend=ax.legend(loc= "upper right", shadow =True, title='ability types')
	plt.xlabel('income levels/ability types')
	plt.ylabel('percentage that received inheritances; 1=%100')
	plt.title('recipiency rate of inheritances with income levels for year {}'.format(str(year)))
	plt.show()





age_income, total_bequests_received, age_income_matrix, total_bequest_matrix = initSCFvars(scf2013, year2013)
age_income_matrix_filled, total_bequest_matrix_filled = genBequestMatrix(age_income_matrix, total_bequest_matrix, \
	age_groups, income_groups, min_age, income_levels, age_income, year2013)
proportion_matrix = genProportionMat(age_groups, income_groups, total_bequest_matrix_filled, total_bequests_received)

age_income2, total_bequests_received2, age_income_matrix2, total_bequest_matrix2 = initSCFvars(scf2010, year2010)
age_income_matrix_filled2, total_bequest_matrix_filled2 = genBequestMatrix(age_income_matrix2, total_bequest_matrix2, \
	age_groups, income_groups, min_age, income_levels, age_income2, year2010)
proportion_matrix2 = genProportionMat(age_groups, income_groups, total_bequest_matrix_filled2, total_bequests_received2)

age_income3, total_bequests_received3, age_income_matrix3, total_bequest_matrix3 = initSCFvars(scf2007, year2007)
age_income_matrix_filled3, total_bequest_matrix_filled3 = genBequestMatrix(age_income_matrix3, total_bequest_matrix3, \
	age_groups, income_groups, min_age, income_levels, age_income3, year2007)
proportion_matrix3 = genProportionMat(age_groups, income_groups, total_bequest_matrix_filled3, total_bequests_received3)

age_income4, total_bequests_received4, age_income_matrix4, total_bequest_matrix4 = initSCFvars(scf2004, year2004)
age_income_matrix_filled4, total_bequest_matrix_filled4 = genBequestMatrix(age_income_matrix4, total_bequest_matrix4, \
	age_groups, income_groups, min_age, income_levels, age_income4, year2004)
proportion_matrix4 = genProportionMat(age_groups, income_groups, total_bequest_matrix_filled4, total_bequests_received4)

age_income5, total_bequests_received5, age_income_matrix5, total_bequest_matrix5 = initSCFvars(scf2001, year2001)
age_income_matrix_filled5, total_bequest_matrix_filled5 = genBequestMatrix(age_income_matrix5, total_bequest_matrix5, \
	age_groups, income_groups, min_age, income_levels, age_income5, year2001)
proportion_matrix5 = genProportionMat(age_groups, income_groups, total_bequest_matrix_filled5, total_bequests_received5)

all_summed_total_bequests = total_bequests_received + total_bequests_received2+ total_bequests_received3+ total_bequests_received4+ total_bequests_received5
all_summed_total_bequests_mat = total_bequest_matrix_filled+ total_bequest_matrix_filled2+ total_bequest_matrix_filled3+ total_bequest_matrix_filled4+ total_bequest_matrix_filled5
all_years_proportion = genProportionMat(age_groups, income_groups, all_summed_total_bequests_mat, all_summed_total_bequests)
# print "here they are: {}\n{}\n{}\n{}\n{}\n{}".format(proportion_matrix, proportion_matrix2, proportion_matrix3, proportion_matrix4, proportion_matrix5, all_years_proportion)


age_income_plot(income_y, age_groups, proportion_matrix, year2013)
age_income_plot(income_y, age_groups, proportion_matrix2, year2010)
age_income_plot(income_y, age_groups, proportion_matrix3, year2007)
age_income_plot(income_y, age_groups, proportion_matrix4, year2004)
age_income_plot(income_y, age_groups, proportion_matrix5, year2001)
age_income_plot(income_y, age_groups, all_years_proportion, year2001_2013)

age_plot(total_bequest_matrix_filled, age_groups, year2013)
age_plot(total_bequest_matrix_filled2, age_groups, year2010)
age_plot(total_bequest_matrix_filled3, age_groups, year2007)
age_plot(total_bequest_matrix_filled4, age_groups, year2004)
age_plot(total_bequest_matrix_filled5, age_groups, year2001)
age_plot(all_summed_total_bequests_mat, age_groups, year2001_2013)

income_plot(total_bequest_matrix_filled, income_y, year2013)
income_plot(total_bequest_matrix_filled2, income_y, year2010)
income_plot(total_bequest_matrix_filled3, income_y, year2007)
income_plot(total_bequest_matrix_filled4, income_y, year2004)
income_plot(total_bequest_matrix_filled5, income_y, year2001)
income_plot(all_summed_total_bequests_mat, income_y, year2001_2013)



