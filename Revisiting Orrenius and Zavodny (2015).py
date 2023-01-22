##########################Replication Paper#################################################################################


# Import packages



import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sans
from numpy.random import seed
import random
import sys
import scipy
import pandas as pd
import os
import seaborn as sans
import math
from scipy import stats
import statsmodels.api as sm
from scipy.stats import ttest_1samp
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed
from numpy.random import randn
from numpy.random import rand
from scipy.stats import norm
from scipy.stats import t as tdist
import matplotlib.transforms as mtrans
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols as olsf
import os
from scipy import stats
from statsmodels.stats.power import TTestIndPower
os.getcwd()
os.chdir("C:/Users/Yussuf Schwarz/OneDrive/Desktop/UniWiwi/703 - Econometrics/Assignments/Replication Paper")
from numpy import *
import us
from stargazer.stargazer import Stargazer

#########################Load Data######################################################

inst = pd.read_csv ("usa_00001.csv.gz")
dat=  pd.read_csv ("usa_00002.csv.gz")
dat=datfig1[dat["BPL"]<950]
dat=datfig1[(dat["AGE"]>22) & (datfig1["AGE"]<65)]
dat["ARRAGE"]=dat["AGE"]-datfig1["YRSUSA1"]
pop= pd.read_excel("Population.xlsx")
inst60 = inst[(inst["YEAR"]==1960)]

#Recoding the State variable and the country of origin variable:
    
fips_name = us.states.mapping('fips', 'name')

state60=[]
for i in inst60["STATEFIP"]:
    if i < 10:
        i=str(i).zfill(2)
        state60.append(fips_name[i])
    else:
        i=str(i)
        state60.append(fips_name[i])
        
origin60=[]
for i in inst60["BPL"]:
    if i < 121:
        origin60.append("USA")
    elif i in (150,160):
        origin60.append("Canada")
    elif i == 200:
        origin60.append("Mexico")
    elif i == 210:
        origin60.append("Central America")
    elif i ==  250:
         origin60.append("Cuba") 
    elif i ==  260:
         origin60.append("Carribean")       
    elif i ==300:
        origin60.append("South America")
    elif i in (410 ,411 , 412 , 413):
        origin60.append("UK")
    elif i == 414:
        origin60.append("Ireland")
    elif i in (400 , 401 , 402 ,403 , 404 , 405 , 420 , 421 , 422 , 423 , 424 , 425 , 426 , 430 , 431 , 432 , 433 , 435 , 436 , 437 ,438 ,439 , 440 , 450 , 451 , 452 , 454 , 456 , 457  ,458 , 459 , 460 , 461 , 462 , 463 , 499):
        origin60.append("Other Europe")
    elif i == 434:
        origin60.append("Italy")
    elif i == 453:
        origin60.append("Germany")
    elif i == 455:
        origin60.append("Poland")
    elif i == 465:
        origin60.append("Russia")
    elif i in range(500,510):
        origin60.append("Other Asia")
    elif i in range(510,520):
        origin60.append("Southeast Asia")
    elif i in range(520,525):
        origin60.append("South Asia")
    elif i in range(530,600):
        origin60.append("Middle East")
    elif i in range(600,1000):
        origin60.append("Rest")
    else:
        print("error")
        
inst60["ORIGIN"]=origin60
inst60["STATE"]=state60


native60=[]
for i in inst60["ORIGIN"]:
    if i == "USA":
        native60.append(1)
    else:
        native60.append(0)
inst60["NATIVE"]=native60


#Immigrant Shares 1960:

c=0
state60=[]
country60=[]
absolute60=pd.DataFrame(inst60["ORIGIN"].unique())
for b in inst60["STATE"].unique():
    c+=1
    state60.append(b)
    absolute=[]
    for i in inst60["ORIGIN"].unique():
        country60.append(i)
        absolute.append(sum((inst60["ORIGIN"]==i) & (inst60["STATE"]==b)))
    absolute60.loc[:,c]=absolute

absolute60.index=absolute60.iloc[:,0]
del absolute60[0]
absolute60.columns=inst60["STATE"].unique()

absolute60.loc[21,:] = absolute60.sum(axis=0)
absolute60.index.values[19]="Sum"

perc60=absolute60
for b in range(51):
    for i in range(20):
        perc60.iloc[i,b]=perc60.iloc[i,b]/perc60.iloc[19,b]
        
        

#Person studied STEM if STEM = 1 
stem=[]
for i in dat["DEGFIELD"]:
    if i in (36,50,24,25,21,37,51):
        stem.append(1)
    else:
        stem.append(0)
dat["STEM"]=stem


#Person is foreign born if BPLD = 1 
bpld=[]
for i in dat["BPL"]:
    if i > 120:
        bpld.append(1)
    else:
        bpld.append(0)
dat["BPLD"]=bpld


#Male if Sex = 0
dat["SEX"]=dat["SEX"]-1

#Person is white, WHITED = 1 
raced=[]
for i in dat["RACE"]:
    if i == 1:
        raced.append(1)
    else:
        raced.append(0)
dat["RACED"]=raced

#Person went to College: colld = 1 
colld=[]
for i in dat["EDUC"]:
    if i > 6:
        colld.append(1)
    else:
        colld.append(0)
dat["COLLD"]=colld

#Year in which the person was 22:
year22=dat["YEAR"]-(dat["AGE"]-22)
dat["YEAR22"]=year22


#########################FIGURE 1######################################################

datfig1=  pd.read_csv ("usa_00002.csv.gz")
datfig1=datfig1[datfig1["BPL"]<950]
datfig1=datfig1[(datfig1["AGE"]>22) & (datfig1["AGE"]<65)]
datfig1["ARRAGE"]=datfig1["AGE"]-datfig1["YRSUSA1"]

#Person studied STEM if STEM = 1 
stem=[]
for i in datfig1["DEGFIELD"]:
    if i in (36,50,24,21,37):
        stem.append(1)
    else:
        stem.append(0)
datfig1["STEM"]=stem

#Year in which the person was 22:
year22=datfig1["YEAR"]-(datfig1["AGE"]-22)
datfig1["YEAR22"]=year22

#Person is foreign born if BPLD = 1 
bpld=[]
for i in datfig1["BPL"]:
    if i > 120:
        bpld.append(1)
    else:
        bpld.append(0)
datfig1["BPLD"]=bpld

#Person went to College: colld = 1 
colld=[]
for i in datfig1["EDUC"]:
    if i > 9:
        colld.append(1)
    else:
        colld.append(0)
datfig1["COLLD"]=colld

shareforeign=[]
year=(sort(datfig1["YEAR22"].unique()))
for i in sort(datfig1["YEAR22"].unique()):
    shareforeign.append((sum((datfig1["ARRAGE"]<20)&(datfig1["BPLD"] ==1) & (datfig1["YEAR22"] == i) & (datfig1["STEM"] == 1) & (datfig1["COLLD"] == 1) )/sum((datfig1["ARRAGE"]<20)& (datfig1["BPLD"] ==1) &(datfig1["YEAR22"] == i)& (datfig1["COLLD"] == 1)))*100)

sharenative=[]
year=[]
for i in sort(datfig1["YEAR22"].unique()):
    year.append(i)
    sharenative.append((sum((datfig1["BPLD"] ==0) & (datfig1["YEAR22"] == i) & (datfig1["STEM"] == 1) & (datfig1["COLLD"] == 1))/sum((datfig1["BPLD"] ==0) & (datfig1["YEAR22"] == i)& (datfig1["COLLD"] == 1)))*100)

plt.plot(year,sharenative, label = "% US natives college graduates majoring in S&E",color="black")
plt.plot(year,shareforeign, label = "% Foreign born college graduates majoring in S&E",linestyle="dashed",color="black")
plt.xlabel('Year / Age22')
plt.ylabel('%')
plt.title('Proportion of college graduates majoring in S&E by nativity')
plt.legend(loc="lower center",bbox_to_anchor=(0.5, -0.35))
plt.show()

#######################################################################################


#########################OLS ESTIMATION TABLE 3######################################################


#Person is in School: Schoold = 1 
schoold=[]
for i in datfig1["EDUC"]:
    if i > 3:
        schoold.append(1)
    else:
        schoold.append(0)
datfig1["SCHOOLD"]=schoold

shareforeignschool=[]
year=(sort(datfig1["YEAR22"].unique()))
for i in sort(datfig1["YEAR22"].unique()):
    shareforeignschool.append((sum((datfig1["ARRAGE"]<20)&(datfig1["BPLD"] ==1) & (datfig1["YEAR22"] == i) & (datfig1["STEM"] == 1) & (datfig1["SCHOOLD"] == 1) )/sum((datfig1["ARRAGE"]<20)& (datfig1["BPLD"] ==1) &(datfig1["YEAR22"] == i)& (datfig1["SCHOOLD"] == 1)))*100)


#%While in college:
shares={"YEAR":year,"SHARE":shareforeign}
shares=pd.DataFrame(shares)

fshare=[]
for i in dat["YEAR22"]:
        d=(shares[shares["YEAR"]==i].iloc[:,1])
        d=np.array(d)
        fshare.append(d[0])



dat["SHARES"]=fshare

#%While in School:
sharesschool={"YEAR":year,"SHARE":shareforeignschool}
sharesschool=pd.DataFrame(sharesschool)

fshares=[]
for i in dat["YEAR22"]:
        d=(sharesschool[sharesschool["YEAR"]==i].iloc[:,1])
        d=np.array(d)
        fshares.append(d[0])


dat["SHARESS"]=fshares

#dat.to_csv("dat.csv", sep=',')
dat=  pd.read_csv ("dat.csv")

datfemale = dat[(dat["SEX"]==2)&(dat["INCWAGE"]>0)]
datmale = dat[(dat["SEX"]==1)&(dat["INCWAGE"]>0)]


olsfemalecollege=smf.ols(formula="STEM~SHARES+AGE+pow(AGE,2)+RACE+C(YEAR)+C(STATEFIP)+BPLD+C(YEAR22)+log(INCWAGE)",data=datfemale).fit()
print(olsfemalecollege.summary())

olsfemaleschool=smf.ols(formula="STEM~SHARESS+AGE+pow(AGE,2)+RACE+C(YEAR)+C(STATEFIP)+BPLD+C(YEAR22)+log(INCWAGE)",data=datfemale).fit()
print(olsfemaleschool.summary())

olsmalecollege=smf.ols(formula="STEM~SHARES+AGE+pow(AGE,2)+RACE+C(YEAR)+C(STATEFIP)+BPLD+C(YEAR22)+log(INCWAGE)",data=datmale).fit()
print(olsmalecollege.summary())

olsmaleschool=smf.ols(formula="STEM~SHARESS+AGE+pow(AGE,2)+RACE+C(YEAR)+C(STATEFIP)+BPLD+C(YEAR22)+log(INCWAGE)",data=datmale).fit()
print(olsmaleschool.summary())

stargazer = Stargazer([olsfemalecollege,olsmalecollege])
stargazer.custom_columns(["Females", "Males"], [1, 1])
print(stargazer.render_latex())

stargazer = Stargazer([olsfemaleschool,olsmaleschool])
stargazer.custom_columns(['Females', 'Males'], [1, 1])
print(stargazer.render_latex())


#######################################################################################

#########################IV ESTIMATION TABLE 4 & 5######################################################

perc60.at["Russia","California"]

fips_name = us.states.mapping('fips', 'name')

state60=[]
for i in dat["STATEFIP"]:
    if i < 10:
        i=str(i).zfill(2)
        state60.append(fips_name[i])
    else:
        i=str(i)
        state60.append(fips_name[i])
        
origin60=[]
for i in dat["BPL"]:
    if i < 121:
        origin60.append("USA")
    elif i in (150,160):
        origin60.append("Canada")
    elif i == 200:
        origin60.append("Mexico")
    elif i == 210:
        origin60.append("Central America")
    elif i ==  250:
         origin60.append("Cuba") 
    elif i in (260,299):
         origin60.append("Carribean")       
    elif i ==300:
        origin60.append("South America")
    elif i in (410 ,411 , 412 , 413):
        origin60.append("UK")
    elif i == 414:
        origin60.append("Ireland")
    elif i in (400 , 401 , 402 ,403 , 404 , 405 , 420 , 421 , 422 , 423 , 424 , 425 , 426 , 430 , 431 , 432 , 433 , 435 , 436 , 437 ,438 ,439 , 440 , 450 , 451 , 452 , 454 , 456 , 457  ,458 , 459 , 460 , 461 , 462 , 463 , 499):
        origin60.append("Other Europe")
    elif i == 434:
        origin60.append("Italy")
    elif i == 453:
        origin60.append("Germany")
    elif i == 455:
        origin60.append("Poland")
    elif i == 465:
        origin60.append("Russia")
    elif i in range(499,510):
        origin60.append("Other Asia")
    elif i in range(510,520):
        origin60.append("Southeast Asia")
    elif i in range(520,525):
        origin60.append("South Asia")
    elif i in range(530,600):
        origin60.append("Middle East")
    elif i in range(600,1000):
        origin60.append("Rest")
    else:
        origin60.append(i)
        

dat["ORIGIN"]=origin60
dat["STATE"]=state60

c=0
year=[]
numberofmigrants=[]
origin=[]
migrants=pd.DataFrame(dat["YEAR22"].unique())

for i in dat["ORIGIN"].unique():
    c+=1
    origin.append(i)
    absolute=[]
    for b in dat["YEAR22"].unique():
        year.append(b)
        absolute.append(sum([(dat["ORIGIN"]==i)&(dat["YEAR22"]==b)]))
    migrants.loc[:,c]=absolute   


migrants.index=migrants.iloc[:,0]
del migrants[0]
migrants.columns=dat["ORIGIN"].unique()



perc60= perc60.drop("USA")
d = np.array( perc60.mean(axis=0))
perc60.loc[18,:] = d
    
absolute60.loc[21,:] = absolute60.sum(axis=0)
perc60.index.values[18]="MEAN"

migrants=migrants.drop("USA", axis=1)
e = np.array( migrants.sum(axis=1))
migrants["SUM"]=e

migrantstates=pd.DataFrame(np.array(migrants.index.values))

c=0
for i in perc60.columns:
    c+=1
    f = np.array(perc60.iloc[18][i]*migrants["SUM"])
    migrantstates.loc[:,c]= f

migrantstates.index=migrantstates.iloc[:,0]
del migrantstates[0]
migrantstates.columns=perc60.columns
del migrantstates["SUM"]

for i in migrantstates.columns:
   migrantstates[i]=((migrantstates[i]/ np.array(pop[pop["NAME"]==i]["CENSUS2010POP"]))*100)

instrumentfinal=[]
for i in range(0,len(dat["STATE"])):
    instrumentfinal.append(migrantstates.at[dat["YEAR22"].iloc[i],dat["STATE"].iloc[i]])

dat["INSTRUMENT"]=instrumentfinal

#1st stage:

datfemale = dat[(dat["SEX"]==2)&(dat["INCWAGE"]>0)]
datmale = dat[(dat["SEX"]==1)&(dat["INCWAGE"]>0)]

modcollege = sm.OLS(dat["SHARES"], dat["INSTRUMENT"])
modcollege = modcollege.fit()
pred1=modcollege.predict(dat["SHARES"])

modschool = sm.OLS(dat["SHARESS"], dat["INSTRUMENT"])
modschool = modschool.fit()
pred2=modschool.predict(dat["SHARESS"])

dat["PREDCOLLEGE"]=pred1
dat["PREDSCHOOL"]=pred2

stargazer = Stargazer([modcollege,modschool])
print(stargazer.render_latex())

#2nd stage:

#dat.to_csv("dat.csv", sep=',')
dat=  pd.read_csv ("dat.csv")

datfemale = dat[(dat["SEX"]==2)&(dat["INCWAGE"]>0)]
datmale = dat[(dat["SEX"]==1)&(dat["INCWAGE"]>0)]

olsfemalecollege=smf.ols(formula="STEM~PREDCOLLEGE+SEX+AGE+pow(AGE,2)+RACE+C(YEAR)+C(STATEFIP)+BPLD+C(YEAR22)+log(INCWAGE)",data=datfemale).fit()
print(olsfemalecollege.summary())

olsfemaleschool=smf.ols(formula="STEM~PREDSCHOOL+SEX+AGE+pow(AGE,2)+RACE+C(YEAR)+C(STATEFIP)+BPLD+C(YEAR22)+log(INCWAGE)",data=datfemale).fit()
print(olsfemaleschool.summary())

olsmalecollege=smf.ols(formula="STEM~PREDCOLLEGE+SEX+AGE+pow(AGE,2)+RACE+C(YEAR)+C(STATEFIP)+BPLD+C(YEAR22)+log(INCWAGE)",data=datmale).fit()
print(olsmalecollege.summary())

olsmaleschool=smf.ols(formula="STEM~PREDSCHOOL+SEX+AGE+pow(AGE,2)+RACE+C(YEAR)+C(STATEFIP)+BPLD+C(YEAR22)+log(INCWAGE)",data=datmale).fit()
print(olsmaleschool.summary())

stargazer = Stargazer([olsfemalecollege,olsmalecollege])
stargazer.custom_columns(["Females", "Males"], [1, 1])
print(stargazer.render_latex())

stargazer = Stargazer([olsfemaleschool,olsmaleschool])
stargazer.custom_columns(['Females', 'Males'], [1, 1])
print(stargazer.render_latex())
