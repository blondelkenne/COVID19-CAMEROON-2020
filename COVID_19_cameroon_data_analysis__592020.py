#!/usr/bin/env python
# coding: utf-8

# # CAMEROON CORONA VIRUS DATA ANALYSIS USING PYTHON

# ##### This Code helps us to understand COVID-19-CAMEROON-Data, data analysis, making decision base on statistical analysis, data analysis technique, machine learning and artificial intelligence.
# #### **An indepth example of SIR time dependent model and SIR travelling wave desease for COVID10 in Cameroon** 
# #### **Other:** Blondel Kenne  
# #### **Company:** kenneelectric llc, **website:** www.kenneelectric.com
# #### **tilte:**  Data scientist  
# #### **date:** 5 - 02 - 2020
# #### **Address**
# #### **Branche 1:** 10130-204 douglas oaks cir Tampa FL 33610 USA
# #### **Branche 2:** 216 Rue Albert Ebe Nlonkak Yaounde-Cameroon
# #### **Tel:** 0018562783591/ **CM:** 653180085
# #### **email:**k.electric@kenneelectric.com 

# # Disclamer

# #### We are not SAR-COV2 0R COVID19 expert and not medical Doctor
# #### We are Data Scientist, Data analysis, Statistician
# #### We analyze data for decision making
# #### We hope this webinar can help you understand Decision bases on data analysis technics
# 

# # **Scenario**
# The following scenario is about COVID19 outbreak in Cameroon as **02-24-2020**.
# 
# Suppose the 10 regions of Cameroon with a population of **26390686**,population density of 52, urban population of **14941523(56.3%)**,median age **19 yrs**, experience an outbreak of the strength of COVID19 that everyone will eventually recover after (**i**) days and which will grants lifelong immunity once recovered.
# 
# When the pandemic was first declared by the government on **02-24-2020**, one person was infected and zero person already immune to the disease.
# 
# Suppose that a susceptible person has a (**p**) chance of contacting a member of the infected population if contact occurs, (**c**) out of every(**k**) people will fall ills with the COVID19
# 

# # What do we know about COVID-19 as of 4/11/2019

# ## All Models are Wrong , Some are useful with uncertainty around

# # I- TIME DEPENDENT MODEL SIR-COVID-19-CAMEROON
# ##### Population Change Over Time

# # Definition and Term
# #### We take all people in Cameroon and divide them into 3 groupes

# # susceptible $S(t)$
# #### All people in Cameroon Capable of becomming sick from the COVID-19
# # Infected $I(t)$
# #### All people in Cameroon infected by the COVID-19(when infected you get it only one time and live the susceptible population
# # Recorvered $R(t)$
# #### All people in Cameroon that recovered or died from COVID-19( can not be reinfected and give to others).Recovered groupe was infected and transition to recovered status to fully recovered or died
# ## $ S(t) + I(t) + R(t)  = N $   (1) ( total population)
# # System of differential equation
# ## we want to know the rate of change of each groupe 
# # $\frac{dS}{dt} $ 
# #### how the number of susceptible is changing over time.Depend on probability of proximity between individual(interaction or likelyhood of interaction
# # $ \frac{dI}{dt} $  
# #### how the number of infection is changing over time. S transit to I to get infected
# # $ \frac{dR}{dt} $
# #### how the number of recorvered is changing over time
# # Reproductive rate or degree of spreadability($R_ro $)
# ### Function of:
# 1-duration of infectivity of affected patient
# 
# 2-Infectiousness of the organism
# 
# 3-number of susceptible population
# 
# 4-environmental condition
# ## Values of R0 of well-known infectious diseases
# #### COVID-19   Respiratory droplets :	3.8–8.9
# 
# 
# ### 5 - Bahavior of the infected population
# it is how much can one person infect others (secondary infection) . it is the number of new cases that existing cases generates on average over the infection perode in the susceptible population.
# 
# 
# it is how much can one person infect (secondary infection) others. it is the number of new cases that existing cases generates on average over the infection perode in the susceptible population.
# ## $ (R_ro = \frac{a S_o}{b} = S_oq) $
# 
# ## $ R_e = \frac{new Cases}{ existing Cases} $ : effective reproductive rate 
# 
# # Serial intervales($S_I$) 
# is defined as the time duration between a primary case-patient (infector) having symptom onset and a secondary case-patient (infectee) having symptom onset (2). The distribution of COVID-19 serial intervals is a critical input for determining the basic reproduction number and the extent of interventions required to control an epidemic(source:https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article) reference:https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3057478/
# 
# # Recovered coefficient (b) :
# It is  1/number of days it takes to revover from the desease(k). 
# 
# $(\frac{1}{k} = b)$ is the mean period of time during which an infected individual can pass it on.
# 
# # Transmition coeficient (a):
# **a**: describes the effective contact rate of the disease: an infected individual comes into contact with n other individuals per unit time (of which the fraction that are susceptible to contracting the disease is $\frac{S}{N} $).
# 
# ## $ a = pq $ 
# 
# where: 
# 
# **p:** probability of becomming in contact with didease
# 
# **q:** chance of falling ills with the disease if you are in contact
# 
# # Assumption to the Model
# In mathematical modeling we have to make assumption to simplify analysis
#  
# ### 1 - Population remain constant(the disease is short)
#  
# ### 2 - Rate of infection proportional to the contact between the susceptible and infected at constant rate
#  
# ### 3- infected will fully recover or died at constant rate 
# 
# 
#  ### An idividual has following stage:
#  ### stage1: Exposed  ($n_E$: number of exposed)
#  An individual is infected has no symptoms and does not yet infect others, the average duration of the exposed stage is $ t_E $
#  ### stage 2: Latent Infection: ($ n_I $: number of latent infection)
#  An individual is infected has no symptoms, but does infect others. the average duration of latent infection is $ t_I $**(Serial intervales($S_I$)**. The individual infect Rro others individual during this period
#  ### stage 3: Isolation
#  After diagnosis, measure are taken to prevent infection (isolating the patient)
#  ## $ R_ro = 1 + K( t_E + t_I)  +  K^2t_Et_I $ (2)
#  
#  where:
#  
#  ## $ K =  \frac {d Ln(I)}{dt} $
#  ### Equation assume that newly infected patient may start transmitting the desease directly after getting infected
#  # Limitation of $R_ro$
#  ### problematic if there are intermediate vector betweens groupe(malaria,typhoid)
#  
#  #### Using COVID19-CAMEROON data,we assume that the median time from onset to clinical recovery for mild cases is approximately 2 weeks(15days) and is 3-6 weeks(42days) for patients with severe or critical disease.
# 
# 

# # MODELLING IN PYTHON 

# # CAMEROON'S CORONA VIRUS DATA  FROM 02-24-2020 T0 05-16-2020
# #### source: https://www.worldometers.info/coronavirus/country/cameroon/;  https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_Cameroon; https://tradingeconomics.com/cameroon/coronavirus-deaths

# In[325]:


# import packages
import math
import scipy.integrate
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime 
# Set ipython's max row display
pd.set_option('display.max_row',50)
# set ipython's max columns display 
pd.set_option('display.max_columns',100)


# In[326]:


# in notebook display
from IPython.display import Image
from IPython.display import display
get_ipython().run_line_magic('matplotlib', 'inline')


# In[327]:


import os
os.getcwd()


# ####   DATA SET 

# In[328]:


# making series from list: series = pd.Series(values) 
date            = pd.Series( pd.date_range(start="2020-02-24",end="2020-05-16") )

daily_deaths    = pd.Series([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,1,1,4,0,0, 
                              0,1,1,1,0,0,0,1,2,0,0,0,0,2,3,0,5,20,0,0,0,1,0,0,10,3,2,0,3,0,0,3,0,0,0,44,0,0,0,6,11,0,11,3,0,1])

Total_deaths     = daily_deaths.cumsum() 

Total_recorvered = pd.Series([0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,0,0,
                              0,0,0,0,0,0,0,0,0,0,5,5,
                              5,5,10,17,17,17,17,60,
                              60,61,98,98,98,108,130,165,165,168,305,305,305,397,397,668,668,700,786,805,915,934,934,934,953,953,953,953,1000,1002,1002,1232,1465,1524,1524,1543,1694,1553,1398])
  
Total_cases      = pd.Series( [1,1,1,1,1,1,1,1,1,1,1,2,
                               2,2,2,2,2,2,2,2,2,2,2,7,
                               7,27,40,0,56,66,70,88,92,113,139,142,
                               193,233,284,509,555,658,658,730,780,803, 
                               820, 820,820, 848, 848, 848,996,996,1011,1017,1017,1163,1163,1334,1430,1518,1621,1705,1705,1832,1832,1832,2077,2077,2104,2104,2265,2267,2267,2274,2579,2689,2689,2800,2954,2954,3105])

Daily_new_cases = pd.Series([1,0,0,0,0,0,0,0,0,0,0,1,
                             0,0,0,0,0,0,0,0,0,0,0,5,
                             0,20,13,0,16,10,4,18,4,21,26,3,
                             51,40,51,225,46,0,103,72,50,73,17,0,
                             0,28,0,0,148,0,17,6,0,146,0,171,96,88,103,84,0,127,0,0,245,0,27,0,161,0,0,7,305,110,0,111,154,0,151])

New_cases_R0 = pd.Series([1,0,0,0,0,0,0,0,0,0,0,5,
                          0,0,0,0,0,0,0,0,0,0,0,20,
                          0,13,17,0,10,4,18,4,21,26,29,51,
                          40,51,225,46,103,0,27,95,73,17,0,0,
                          28,0,0,148,0,17,6,0,146,0,171,96,88,103,84,0,127,0,0,245,0,27,0,161,0,0,7,305,110,0,111,154,0,151,0])

Active_cases     = Total_cases - (Total_recorvered + Total_deaths)

# Data frame: Covid_19_cameroon_data

Covid_19_cameroon_data = pd.DataFrame( {'date':date, 'Total_cases':Total_cases,'Daily_new_cases':Daily_new_cases, 'New_cases_R0':New_cases_R0,'Active_cases':Active_cases,'Total_deaths':Total_deaths,
                                         'daily_deaths':daily_deaths,'Total_recorvered':Total_recorvered})

Covid_19_cameroon_data['Reproductive_ratio'] = (Covid_19_cameroon_data['New_cases_R0']/Covid_19_cameroon_data['Active_cases']).fillna(0)

# write the Covid_19_cameroon_data to csv format 
Covid_19_cameroon_data.to_csv('Covid_19_cameroon_data.csv')


# In[329]:


x1 = (Covid_19_cameroon_data["Daily_new_cases"].replace(0, pd.np.nan).dropna(axis=0, how='any')).values
log_dailyNewCases = pd.DataFrame({'log_dailyNewCases': x1})
log_dailyNewCases  = np.log(log_dailyNewCases).mean().values[0] 


# In[330]:


log_dailyNewCases


# In[331]:


# view data set
Covid_19_cameroon_data


# ### MODEL
# ##### The SIR model differential equations.
# # $\frac{dS}{dt} = - aIS $  (3)
# # $\frac{dI}{dt} = aIS - bI$ (4)
# # $\frac{dR}{dt} = bI$  (5)
# 
# ###  $ I(t) $after the initial condition
# ## $ \frac {dI}{dt} = I(aS_o - b) $ (6)
# ## $ I(t) = \exp(aS_o - b) $ (7)
# 

# In[332]:



def SIR_TIME_DEPENDENT_MODEL(Y,t,a,b):
    S,I,R = Y
    dS_dt = -a*S*I
    dI_dt = a*S*I - b*I
    dR_dt = b*I
 
    return([dS_dt,dI_dt,dR_dt])


# # Reproductive rate or degree of spraidability($R_e$): 
# ### how much can one person infect (secondary infection) others.
# ### the number of new cases that existing cases generates on average over the infection perode in the susceptible population.
# ### Values over 1.0 mean we should expect more cases in Cameroon, values under 1.0 mean we should expect fewer.

# In[333]:


import statistics
from scipy.stats import skew
x = Covid_19_cameroon_data["Reproductive_ratio"]
x1 = (Covid_19_cameroon_data["Reproductive_ratio"].replace(0, pd.np.nan).dropna(axis=0, how='any')).values
x1 = pd.DataFrame({'Reproductive_ratio_Wout_Zero': x1})
x11 = x1['Reproductive_ratio_Wout_Zero']


# # $ R_e $ mean with  $ {R_e}/days = 0 $

# In[334]:



# Measures of Central Tendency
mean1  = x.mean()
median = x.median()
mode   = x.mode()
sns.set(style="whitegrid")
## box plot R0
ax = sns.boxplot(x=Covid_19_cameroon_data["Reproductive_ratio"])
print('Mean: {}'.format(mean1))
print('Median: {}'.format(median))
print('Mode: {}'.format(mode))
print('max: ',x.max())
print('min: ',x.min())
print('Skewness: {}'.format(skew(x)))
#Measures of Dispersion
r=x.max()-x.min()
print('Range: ',r)
#Variance
sigma=np.std(x)
print('Standard Deviation: ',sigma)


# In[335]:


# plot R0
plt.figure(figsize = [12,8])
t= Covid_19_cameroon_data['date']
y= Covid_19_cameroon_data['Reproductive_ratio']
# Plot the data on three separate curves for S(t), I(t) and R(t)
plt.figure(figsize = [12,8])
plt.plot(t, y,label = "Rr0:Reproductive ration")
xcoords = [1, 0.2, 2.85]
# colors for the lines
colors = ['r','k','b']

for xc,c in zip(xcoords,colors):
    plt.axhline(y=xc, label='Reproductive Ratio at Re = {}'.format(xc), c=c)

plt.legend()
plt.xlabel("date")
plt.ylabel("Reproductive ratio(degree of spraidability)")
plt.title("T.D Model:SIR-COVID-19 CAMEROON ** 02/24-04/28-2020 ** Reproductive ration(Re)")
plt.show


# # $ R_e $  mean without $ R_e/days = 0 $  

# In[336]:



# Measures of Central Tendency
mean2   = x11.mean()
median = x11.median()
mode   = x11.mode()
sns.set(style="whitegrid")
## box plot R0
ax = sns.boxplot(x=x1)
print('Mean: {}'.format(mean2))
print('Median: {}'.format(median))
print('Mode: {}'.format(mode))
print('max: ',x11.max())
print('min: ',x11.min())
print('Skewness: {}'.format(skew(x11)))
#Measures of Dispersion
r=x11.max()-x11.min()
print('Range: ',r)
#Variance
sigma=np.std(x11)
print('Standard Deviation: ',sigma)


# In[337]:


# plot R0
plt.figure(figsize = [12,8])
t= x1.index 
y= x1['Reproductive_ratio_Wout_Zero'] 
# Plot the data on three separate curves for S(t), I(t) and R(t)
plt.figure(figsize = [12,8])
plt.plot(t, y,label = "Rr0:Reproductive ration")
xcoords = [1, 0.2, 2.85]
# colors for the lines
colors = ['r','k','b']

for xc,c in zip(xcoords,colors):
    plt.axhline(y=xc, label='Reproductive Ratio at Re = {}'.format(xc), c=c)

plt.legend()
plt.xlabel("date")
plt.ylabel("Reproductive ratio(degree of spraidability)")
plt.title("T.D Model:SIR-COVID-19 CAMEROON ** 02/24-04/16-2020 ** Reproductive ration(Re)")
plt.show


# # Serial intervales ($ S_I $) :
# ### Time duration between a primary case-patient (infector) having symptom onset and a secondary case-patient (infectee) having symptom onset (2). 
# ### Transmission events reported during the early stages of outbreaks
# column = ["Event_index","Infector_ID","Infectee ID","City","Province","infection_location","Infector_symptom_onset_date","Age",
# "Sex","Secondary_infection location","Seconday_symptom onset date","Seconday_Age","Seconday_Sex","Contact_type",]

# # INITIAL CONDITION 
# ## $ S(0) = S_o $
# ## $ I(0) = I_0 $
# ## $ R(0) = R_0 $
# ## $ \frac {d(S + I + R)}{dt} = 0 $
# ## $S + I + R = S_o + I_o $
# ## BEFORE THE GOVERMENT MESURES 
# ### Initial condition date: 03-15-2020

# In[338]:


# Total population, N.
N = 26390686 
# Initial number of infected and recovered individuals, I0 and R0.
# as of the first declare case in cameroon on 02-24-2020 to 03-20-2020
# Everyone else, S0, is susceptible to infection initially.
I0,R0 = 27, 0  # on 03-20-2020
S0 = N - I0 - R0
# b is 1/number of days it takes to revover from the desease(n)  
# 1/n is the mean period of time during which an infected individual can pass it on.
# we assume it takes in average 15-30 days to recover from Covid-19 in Cameroon
b =  1 / 15
# transmition coefficient with Rr0( daily reproduction rate of COVID-19 cases from 02-24-2020 t0 03-20-2020, Rr0_min = 0
Rr0 = [1.27,2.11,2.85]
#Rr0_av_wthZerodays = 1.27
a0 = ( Rr0[0] * b) / S0
#Rro_av_woutZerodays = 2.11
a1 = ( Rr0[1] * b) / S0
#Rr0_max = 2.85
a2 =  ( Rr0[2] * b) / S0


# # TIME VICTOR

# In[339]:


# A grid of time points (in days)
#numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
#Return evenly spaced numbers over a specified interval.
#Returns num evenly spaced samples, calculated over the interval [start, stop].
#The endpoint of the interval can optionally be excluded.
#Changed in version 1.16.0: Non-scalar start and stop are now supported.

t = np.linspace(0, 160, 160)


# #1- population of various compartment over 6 months and Model of their behavior graphically
# # Predict the future $ S(t) $,$ I(t) $,$ R(t) $ before the gorverment measure

# In[340]:


# Initial conditions vector
# Integrate the SIR equations over the time grid, t.
solution0 = scipy.integrate.odeint( SIR_TIME_DEPENDENT_MODEL,[S0,I0,R0],t,args = (a0,b))
soluion0  = np.array(solution0)

solution1 = scipy.integrate.odeint( SIR_TIME_DEPENDENT_MODEL,[S0,I0,R0],t,args = (a1,b))
soluion1  = np.array(solution1)

solution2 = scipy.integrate.odeint( SIR_TIME_DEPENDENT_MODEL,[S0,I0,R0],t,args = (a2,b))
soluion2  = np.array(solution2)


# # Plot the future  $ S(t) $,$ I(t) $,$ R(t) $

# In[341]:


# Plot the data on three separate curves for S(t), I(t) and R(t)
plt.figure(figsize = [10,8])
plt.plot(t, solution0[:,0]/10000,label = "S(t):Susceptible befor gov mesures")
plt.plot(t, solution0[:,1]/10000,label = "I(t):Infected before gov mesures")
plt.plot(t, solution0[:,2]/1000,label = "R(t):Recovered with immunity")
plt.grid()
plt.legend()
plt.xlabel("Time/1days")
plt.ylabel("Population Number per 10000s")
plt.title("Time Dependent Model:SIR-COVID-19 CAMEROON ** 02/24-03/20-2020 ** Rr0=1.27 ** before gov mesure on 03-15-2020")
plt.show


# In[342]:


# Plot the data on three separate curves for S(t), I(t) and R(t)
plt.figure(figsize = [10,8])
plt.plot(t, solution1[:,0]/10000,label = "S(t):Susceptible before gov mesures")
plt.plot(t, solution1[:,1]/10000,label = "I(t):Infected before gov mesures")
plt.plot(t, solution1[:,2]/10000,label = "R(t):Recovered with immunity mesures")
plt.grid()
plt.legend()
plt.xlabel("Time/1days")
plt.ylabel("Population Number per 10000s")
plt.title("Time Dependent Model:SIR-COVID-19 CAMEROON ** 02/24-03/20-2020 ** Rr0=2.11 ** before gov mesure on 03-15-2020")
plt.show


# In[343]:


# Plot the data on three separate curves for S(t), I(t) and R(t)
plt.figure(figsize = [10,8])
plt.plot(t, solution2[:,0]/10000,label = "S(t):Susceptible before gov mesures")
plt.plot(t, solution2[:,1]/10000,label = "I(t):Infected before gov mesures")
plt.plot(t, solution2[:,2]/10000,label = "R(t):Recovered with immunity befor gov mesures")
plt.grid()
plt.legend()
plt.xlabel("Time/1days")
plt.ylabel("Population Number per 10000s")
plt.title("Time Dependent Model:SIR-COVID-19 CAMEROON ** 02/24-03/20-2020 ** Rr0=2.85 ** before gov mesure on 03-15-2020")
plt.show


# # Create Dataframe from the future $ S(t) $ ,$ I(t) $,$ R(t) $

# In[344]:


Susceptible_population =  pd.Series(solution2[:,0])
Infected_population    =  pd.Series(solution2[:,1])
Removed_population     =  pd.Series(solution2[:,2])
date                   = pd.Series( pd.date_range(start="2020-04-17",end="2020-09-23") )

SIR_MODEL_COVID19_CAMEROON = pd.DataFrame( {'date':date, 'Susceptible_population':Susceptible_population, 'Infected_population':Infected_population, 'Removed_population':Removed_population})
SIR_MODEL_COVID19_CAMEROON.to_csv('SIR_MODEL_COVID19_CAMEROON_BEFOR_GOV_MSR.csv')
max_Infected = SIR_MODEL_COVID19_CAMEROON[SIR_MODEL_COVID19_CAMEROON['Infected_population'] == SIR_MODEL_COVID19_CAMEROON['Infected_population'].max() ]
max_removed  = SIR_MODEL_COVID19_CAMEROON[SIR_MODEL_COVID19_CAMEROON['Removed_population'] == SIR_MODEL_COVID19_CAMEROON['Removed_population'].max() ]
SIR_MODEL_COVID19_CAMEROON


# In[345]:


max_Infected


# In[346]:


max_removed


# # Pair plot the future $ S(t) $,$ I(t) $,$ R(t) $

# In[347]:


g = sns.pairplot(SIR_MODEL_COVID19_CAMEROON, diag_kind="kde")


# # Joint plot S(t) and I(t)

# In[348]:


ax = sns.scatterplot(x=SIR_MODEL_COVID19_CAMEROON.index, y="Infected_population", data=SIR_MODEL_COVID19_CAMEROON)


# In[349]:


ax = sns.scatterplot(x=SIR_MODEL_COVID19_CAMEROON.index, y="Removed_population", data=SIR_MODEL_COVID19_CAMEROON)


# In[350]:


ax = sns.scatterplot(x=SIR_MODEL_COVID19_CAMEROON.index, y="Susceptible_population", data=SIR_MODEL_COVID19_CAMEROON)


# # INITIAL CONDITION AFTER THE GOVERMENT MEASURES 
# 

# ####  INITIAL CONDITION DATE: 03-15-2020  #### 
# #### Cameroon Coronavirus Cases:848 Deaths:17 Recovered:165  source:http://apanews.net/en/pays/cameroun/news/cameroons-health-ministry-stops-daily-covid-19-reports####

# In[351]:


# Total population, N.
N = 26390686 
# Initial number of infected and recovered individuals, I0 and R0.
# as of the last declare case in cameroon on 04-15-2020
# Everyone else, S0, is susceptible to infection initially.
# b is 1/number of days it takes to revover from the desease(n)  
# 1/n is the mean period of time during which an infected individual can pass it on.
# we assume it takes in average 15 days to recover from Covid-19 in Cameroon
b =  1/15
# on 04-28-2020 ## source: https://www.worldometers.info/coronavirus/country/cameroon/
I0 = max(Covid_19_cameroon_data['Total_cases'])
R0 = max(Covid_19_cameroon_data['Total_deaths'] + Covid_19_cameroon_data['Total_recorvered'] )
S0 = N - I0 - R0
# transmition coefficient with Rr0( daily reproduction rate of COVID-19 cases from 02-24-2020 t0 03-20-2020, Rr0_min = 0
Rr0 = [mean2,mean1]
#Rr0_av_wthZerodays 
a3 =  ( Rr0[0] * b) / S0
#Rro_av_woutZerodays = 
a4 =  ( Rr0[1] * b) / S0


# In[373]:


I0


# # Result
# # Predict the future $ S(t),I(t),R(t) $ after the gorverment measure

# In[352]:


# Initial conditions vector
# Integrate the SIR equations over the time grid, t.
solution3 = scipy.integrate.odeint( SIR_TIME_DEPENDENT_MODEL,[S0,I0,R0],t,args = (a3,b))
soluion3  = np.array(solution3)

solution4 = scipy.integrate.odeint( SIR_TIME_DEPENDENT_MODEL,[S0,I0,R0],t,args = (a4,b))
soluion4  = np.array(solution4)


# # Plot the future $ S(t),I(t),R(t) $

# In[353]:


# Plot the data on three separate curves for S(t), I(t) and R(t)
plt.figure(figsize = [10,8])
plt.plot(t, solution3[:,0]/10000,label = "S(t):Susceptible after gov mesures")
plt.plot(t, solution3[:,1]/10000,label = "I(t):Infected after gov mesures")
plt.plot(t, solution3[:,2]/10000,label = "R(t):Recovered with immunity after gov mesures")
plt.grid()
plt.legend()
plt.xlabel("Time /1days")
plt.ylabel("Population Number per 10000s")
plt.title('Time Dependent Model:SIR-COVID-19 CAMEROON ** 02/24-03/20-2020 **after gov mesures ** 04-15-2020 ** Rr0: {}'.format(mean2))
plt.show


# # Create Dataframe from the future $ S(t),I(t),R(t) $

# In[354]:


Susceptible_population =  pd.Series(solution3[:,0])
Infected_population    =  pd.Series(solution3[:,1])
Removed_population     =  pd.Series(solution3[:,2])
date                   = pd.Series( pd.date_range(start="2020-04-17",end="2020-09-23") )

SIR_MODEL_COVID19_CAMEROON = pd.DataFrame( {'date':date, 'Susceptible_population':Susceptible_population, 'Infected_population':Infected_population, 'Removed_population':Removed_population})
SIR_MODEL_COVID19_CAMEROON.to_csv('SIR_MODEL_COVID19_CAMEROON_AFTER_GOV_MSR.csv')
max_Infected1 = SIR_MODEL_COVID19_CAMEROON[SIR_MODEL_COVID19_CAMEROON['Infected_population'] == SIR_MODEL_COVID19_CAMEROON['Infected_population'].max() ]
max_removed1  = SIR_MODEL_COVID19_CAMEROON[SIR_MODEL_COVID19_CAMEROON['Removed_population'] == SIR_MODEL_COVID19_CAMEROON['Removed_population'].max() ]
SIR_MODEL_COVID19_CAMEROON


# In[355]:


max_Infected1['Removed_population'].values[0]


# In[356]:


max_removed1


# In[357]:


g = sns.pairplot(SIR_MODEL_COVID19_CAMEROON, diag_kind="kde")


# In[358]:


ax = sns.scatterplot(x=SIR_MODEL_COVID19_CAMEROON.index, y="Infected_population", data=SIR_MODEL_COVID19_CAMEROON)


# In[359]:


ax = sns.scatterplot(x=SIR_MODEL_COVID19_CAMEROON.index, y="Removed_population", data=SIR_MODEL_COVID19_CAMEROON)


# In[360]:


ax = sns.scatterplot(x=SIR_MODEL_COVID19_CAMEROON.index, y="Susceptible_population", data=SIR_MODEL_COVID19_CAMEROON)


# # Plot the future  $ S(t),I(t),R(t) $
# 
# 
# 

# In[361]:


# Plot the data on three separate curves for S(t), I(t) and R(t)
plt.figure(figsize = [10,8])
plt.plot(t, solution4[:,0]/10000,label = "S(t):Susceptible after gov mesures")
plt.plot(t, solution4[:,1]/10000,label = "I(t):Infected after gov mesures")
plt.plot(t, solution4[:,2]/10000,label = "R(t):Recovered with immunity after gov mesures")
plt.grid()
plt.legend()
plt.xlabel("Time /1days")
plt.ylabel("Population Number per 10000s")
plt.title('Time Dependent Model:SIR-COVID-19 CAMEROON ** 02/24-03/20-2020 **after gov mesures ** 04-15-2020 ** Rr0: {}'.format(mean1))
plt.show


# In[362]:


Susceptible_population =  pd.Series(solution4[:,0])
Infected_population    =  pd.Series(solution4[:,1])
Removed_population     =  pd.Series(solution4[:,2])
date                   = pd.Series( pd.date_range(start="2020-04-17",end="2020-09-23") )

SIR_MODEL_COVID19_CAMEROON = pd.DataFrame( {'date':date, 'Susceptible_population':Susceptible_population, 'Infected_population':Infected_population, 'Removed_population':Removed_population})
SIR_MODEL_COVID19_CAMEROON.to_csv('SIR_MODEL_COVID19_CAMEROON_AFTER_GOV_MSR_Rr.csv')
SIR_MODEL_COVID19_CAMEROON


# In[363]:


g = sns.pairplot(SIR_MODEL_COVID19_CAMEROON, diag_kind="kde")


# # 2- Will the COVID_19 spread in Cameroon

# ### We know that Susceptible (S) always less than :$ S_o ->(S < S_o)-> S < 26390686 $
# ## $ \frac{dI}{dt} < I(aS_o - b) => S_o > \frac{b}{a}  $
# if we change the equation we also get 
# # $ R_ro =\frac{ a  S_o} { b} =  qS_o > 1 $
# # For COVID_19 to spread the reproduction ratio should be greather than one ($ R_ro > 1 $) 
# ### According to the data collected from 02-24-2020 T0 05-01-2020 on COVID-19 in Cameroon, The avarage reproduction ration is  between  $( 0.176 <= R_e <= 0.346 ) $
# # We can conlude that: in cameroon according to public available data, the COVID-19 will not spread 

# # 3- What is the maximum infected population at any time
# ###  min ,max => differential the function , but we already have a diff equation (1),(2),(3)

# ## $\frac{ \frac{dI}{dt}}{\frac{dS}{dt}} => \frac{dI}{dS} = \frac{aSI - bI}{-aSI }= -1 + \frac{1}{qS} $
# 
# ## $I$ max will accure in 
# # $ I + S - \frac{1}{q}ln(S) = I_o + S_o - \frac{1}{q}ln(S_o) $(8)
# when $ S = \frac{1}{q}$  $ \frac {dI}{dS} = 0 $ , we substitut in the equation above to get I as Imax
# # $I_max = I_o + S_o - \frac{1}{q}(1 + ln(qS_o)) $
# 
# # $ f(x) = - \frac{1}{q }(1 + ln(qS_o)) $

# ## Predict the future $ I =  f(S) $ before the gorverment measure

# In[364]:


# Plot the data on three separate curves for S(t), I(t) and R(t)


xmax = max_Infected['Susceptible_population'].values[0]
ymax = max_Infected['Infected_population'].values[0]
plt.figure(figsize = [8,8])
plt.plot(solution2[:,0], solution2[:,1],label = " I = f(S)")


plt.annotate('Predicted I_max at S = 9.046548e+06,t = 2020-08-12 ', xy=(xmax, ymax), xytext=(9000000,6000000),
             arrowprops=dict(facecolor='black', shrink=0.5),)
plt.grid()
plt.legend()
plt.xlabel("Predicted Susceptible population per 10000")
plt.ylabel("predicted Infected Population per 10000s")
plt.title("SIR-COVID-19 CAMEROON **Re=2.85 ** I = f(s) before gov mesure")
plt.show

############################################################################################################################

x = pd.Series(solution2[:,0])
y = pd.Series(solution2[:,1])
fig, ax = plt.subplots()
ax.plot(x,y)

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.6, 0.8), **kw)

annot_max(x,y)


ax.set_ylim( 0,10000000)
plt.show()


# ## Predict the future $ I =  f(S) $ after the gorverment measure

# In[365]:


# Plot the data on three separate curves for S(t), I(t) and R(t)

xmax = max_Infected1['Susceptible_population'].values[0]
ymax = max_Infected1['Infected_population'].values[0]
plt.figure(figsize = [8,8])
plt.plot(solution3[:,0], solution3[:,1],label = " I = f(S)")


plt.annotate('Predicted I_max at S = 26386348,t = 2020-04-17 ', xy=(xmax, ymax),#xytext=(),
             arrowprops=dict(facecolor='black', shrink=0.5),)
plt.grid()
plt.legend()
plt.xlabel("Predicted Susceptible population per 10000")
plt.ylabel("predicted Infected Population per 10000s")
plt.title("SIR-COVID-19 CAMEROON **Re ** I = f(s) after gov measure")
plt.show

############################################################################################################################

x = pd.Series(solution3[:,0])
y = pd.Series(solution3[:,1])
fig, ax = plt.subplots()
ax.plot(x,y)

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.6, 0.8), **kw)

annot_max(x,y)


ax.set_ylim( 0,10000)
plt.show()


# # 4- How many people will catch COVID-19 in Cameroon

# ### We want the COVID-19 to end at $ dI/dt = 0 = end ===> S + I + R = S0 + I0 $ 
# we go back to (8) and let time progress to the end of the pandemic
# 
# ## $ R_{end} = -S_{end} + S0 + I0 $  where $ R_{end} = death + recovered $ 
# 
# 
# ## $ S_{end}  - \frac{1}{q}ln(S_{end}) = S_0 +I_0 - \frac{1}{q}ln(S_o) $ where **q: Contact ratio**
# 
# we substitute Send in Rend ,solve the equation, change variable to get 
# 
# # $ Y - \frac{1}{q}ln(y) = C $ 
# 
# 

# ## Predict the future $ R = f(S) $ before the gorverment measure

# In[366]:



# Plot the data on three separate curves for S(t), I(t) and R(t)
xmax = max_removed['Susceptible_population'].values[0]
ymax = max_removed['Removed_population'].values[0]
plt.figure(figsize = [8,8])
plt.plot(solution2[:,0], solution2[:,2],label = " I = f(S)")


plt.annotate('Predicted R_max at S =2.337462e+06 ,t = 2020-09-23 ', xy=(xmax, ymax), #xytext=,
             arrowprops=dict(facecolor='black', shrink=0.5),)
plt.grid()
plt.legend()
plt.xlabel("Predicted Susceptible population")
plt.ylabel("predicted Removed Population ")
plt.title("SIR-COVID-19 CAMEROON **Rr0=2.85 ** R = f(s) before gov mesure")
plt.show

############################################################################################################################

x = pd.Series(solution2[:,0])
y = pd.Series(solution2[:,2])
fig, ax = plt.subplots()
ax.plot(x,y)

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.6,0.8 ), **kw)

annot_max(x,y)


ax.set_ylim( 0,30000000)
plt.show()




# ## Predict the future $ R =  f(S) $ before the gorverment measure

# In[367]:



# Plot the data on three separate curves for S(t), I(t) and R(t)
xmax = max_removed1['Susceptible_population'].values[0]
ymax = max_removed1['Removed_population'].values[0]
plt.figure(figsize = [8,8])
plt.plot(solution3[:,0], solution3[:,2],label = " I = f(S)")


plt.annotate('Predicted R_max at S =2.337462e+06 ,t = 2020-09-23 ', xy=(xmax, ymax), #xytext=(),
             arrowprops=dict(facecolor='black', shrink=0.5),)
plt.grid()
plt.legend()
plt.xlabel("Predicted Susceptible population")
plt.ylabel("predicted Removed Population ")
plt.title("SIR-COVID-19 CAMEROON **Re ** R = f(s) after gov measure")
plt.show

############################################################################################################################

x = pd.Series(solution3[:,0])
y = pd.Series(solution3[:,2])
fig, ax = plt.subplots()
ax.plot(x,y)

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.6,0.8 ), **kw)

annot_max(x,y)


ax.set_ylim( 0,6000)
plt.show()


# # 5- infection threshold
# ### The number of susceptible population for infection rate to be zero(changing from a positive infection to negative infection)
# $ dI/dt = 0 $

# In[368]:


print ("Infection threshold for Rr0 = 1.27 is :",b/a0)
print ("Infection threshold for Rr0 = 2.11  is :",b/a1)
print ("Infection threshold for Rr0 = 2.85 is :",b/a2)


# # To significantly slow the spread of the disease (dI/dt < 0) after DAY(0) Governemt measure was put in place to reduce the chance of contacting the disease(COVID-19) by factor of (k)
# 
# #6-How those the threshold change in this situation

# In[369]:


# a_new: new transmition coefficient
# b_new: new recovered rate(do not change most of the time with preventive mesure)
k = 2.5
print("the new threshold to reduce the chance of contacting COVID-19-Rr = 1.27 by factor of 20 is:", k * (b/a0))
print("the new threshold to reduce the chance of contacting COVID-19-Rr = 2.11 by factor of 20 is:", k * (b/a1))
print("the new threshold to reduce the chance of contacting COVID-19-Rr = 2.85 by factor of 20 is:", k * (b/a2))


# # what all this mean for COVID19 cameroon
# ## Spread of the virus in Cameroon** : $ R_ro = qS_o > 1 $ 
# ## Maximun infected populations :$ Total population -  f(q) $
# ## How many people will catch the COVID-19 in Cameroon:$ Total population - g(q) $
# ## **q**: contact ration is very important **
# ### **Series Intervals ($ S_I $)**: From the start of symptom of the **infector** to the start of symptom of **Infectee**  it takes how many days
# ### Exemple $ R_ro, S_I $
# ### $ R_ro $:1 person can infect 3 persons( $ R_ro = 3 $)
# ### $ S_I $: on person can spread the virus to others persons in 7 days ($ S_I=7 $ )
# # Recommendation
# ## 1 - to decrease the probability of contact and the chance of becoming hill: Wash hands, sneeze into elbow, keep sick kids home ,close school ...etc
# 
# ## 2- to decrease the susceptible population: Vaccination
# 
# ## 3- to increase the removed population: antibiotic(rarely effective)
# 
# ## 4- Wait COVID-19 eventualy run out of Susceptibles 

# # II- SIR TRAVELLING WAVE CAVID-19 MODEL- CAMEROON(Population Change over Space)

# # Assumption to the Model
# 1 - susceptible do not move
# 
# 2 - Infected  migrate at constant rate
# 
# 3 - Recorvered or Removed population do not move
# 
#  
# # Travelling Wave Model (the population depend on time and space) 
# # $ \frac{\partial dS}{\partial dt} = -aIS $ (9) 
# # $ \frac{\partial dI}{\partial dt}  = aIS -aI +D\frac{\partial d^2I}{\partial dt^2} $  (10)
# # $ \frac{\partial dR}{\partial dt} = -aI $ (11)
# 
# ### $ \frac{\partial dS}{\partial dt}$,$ \frac{\partial dI}{\partial dt} $, $ \frac{\partial dR}{\partial dt} $ depend both on time and spaces 
# 
# ### D : constant rate of diffussion
# non dimentionalization : Combine  all constant together to have a key parameter =  $ R_o $ 
# 
# convert equation into differential equation by changing variable
# 
# create a new variable called **y** and convert equation into differencial equation that involve only space variable called **x**
# 
# ### $ y = x - ct $
# 
# (9) (10) (11) become
# 
# # $ 0 = C\frac {dS}{dy} - IS $ (12)
# # $ 0 = \frac{d^2S}{dy^2} + C\frac{dI}{dy}+ I(S - \frac{1}{R_o}) $ (13)
# we ignore  R
# ### WAVE MODEL IS FOCUST ON $ I(t) $

# # Initial condition

# ### Before the start of the pandemic : we go backword in time ( $ +\infty <-- \infty $)  $ <-- t $ 
# #### As (14)
# ####   $  y --> +\infty (pass) $
# ####   $  I --> O $
# ####   $  S --> 1 $
# 
# 
# ### We can expect what will happen in the futur : we let t --> $ +\infty $
# ### As (15)
# #### $ y --> -\infty (futur) $
# #### $ I --> 0 $
# #### $ S? $
# 

# # 7- WHAT IS THE WAVE SPEED OF COVID-19 IN CAMEROON UP TO 04-15-2020
# Linearization: we use the value of S in the pass to make approximation **p** 
# #### (14) --> $ S = 1-P $  substitut into (12)  (13)
# 
# # $  -C\frac {dP}{dy} - I = 0 $ (16)
# # $   \frac{d^2I}{dy^2} + C\frac{dI}{dy}+ I(1 - \frac{1}{R_o}) = 0 $ (17)
# we use another tools phase plane analysis to get the travel wave solution.
# For the solution to exist, the minimum require wave speed is $ 2\sqrt{1-\frac{1}{R_ro}} $
# 
# where 
# ### $ R_{ro} = S_o q $ 
# with 
# ### $ q = \frac{a}{ b }$
# 
# # $ Cmin = 2 \sqrt{1-\frac{1}{R_ro}} $
# ### C is the speed of travelling wave also the speed of COVID-19 as it propagate throught the population. For COVID-19, the speed of spraid C should be minimum as posible to slow dow the propagation of the virus throught the population. We can make C small by making  R0 small  PROVIDED THAT $ R_ro > 1 $
# 

# In[370]:



Rr0 = mean2 
Speed_of_the_spraid_of_COVID19 = 2 * np.sqrt( 1 - (1/Rr0 ))

if (Rr0 < 1):
    print("the speed of the spraid of COVID-19 in CAMEROON is : do not exit because Rro < 1 ")
else :
    print (" the speed of the spraid of COVID-19 in CAMEROON is : ",Speed_of_the_spraid_of_COVID19)


# # 8- WHAT IS THE SEVERITY OF COVID-19 IN CAMEROON AS OF 04-15-2020 
# If we get how many people was left after the disease we can get how many was actualy affected : S at initial condition (14)
# 
# we substitut (12) into (13) and integrate to get : 
# # $ \frac{dI}{dy} + CI+ C(S - \frac{1}{R_ro}ln(S))= C $ (16)
# #### with  initial condition (14) $ \frac{dI}{dy} \approx 0 , CI \approx 0,   S = 1 $ 
#  
# 
# to calculate C  we used the initial condition (15)  $ \frac{dI}{dy} \approx 0 , CI \approx 0,   S = S_{end} $ 
# # $ S_{end} - \frac{1}{R_ro}  ln (S_{end}) = 1 $ 
# S is the proportion of the initial population
# 
# 
# 
# 
# #### The best way to deal with implicit equation is to plot a gragh to find the relationship between $S_{end}$   $R_ro $ 
# 
# $ S_{end} = f(R_ro) $
# 

# In[371]:


data = { 'Send': [1, 0.8, 0.6,0.2, 0.1, 0.05, 0.03, 0.01, 0],'Rr0': [1, 1.25,  1.35, 1.50, 1.75, 2.25, 2.75, 2.85, 3] }
data = pd.DataFrame(data)
data


# In[372]:


plt.figure(figsize = [10,8])
plt.plot(data['Rr0'], data['Send'],label = "Send = f(Rr0)")
plt.grid()
plt.legend()
plt.xlabel("Reproductive Ratio")
plt.ylabel("Susceptible range of Value")
plt.title("Severity of COVID 19: number of Susceptible left at the end of COVID-19 ")
plt.show


# ## Model are forecast , not our fate and our bihavior can change the forecast in a big way

# # **conclusion**
# 
# ## In Cameroon, the data collected on the first 2077 people diagnosed with COVID19 as of 04-30-2020 show that, the pandemic did not start yet because the reproductive ratio(Rro) is less than one (**Rr0 < 1**) and the transmission coefficient or contact ratio is very small( **a** is very small). 
# 
# ## We can conclud that the Goverment Measures put in place on 03-15-2020 to lower the tansmission coefficient or contact ratio to significantly slow down COVID19 works according to the public available data published by the Cameroon's minister of health   
