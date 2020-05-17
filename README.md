CAMEROON CORONA VIRUS DATA ANALYSIS USING PYTHON

This Code helps us to understand COVID-19-CAMEROON-Data, data analysis, making decision base on statistical analysis, data analysis technique, machine learning and artificial intelligence.
An indepth example of SIR time dependent model and SIR travelling wave desease for COVID10 in Cameroon
Other: Blondel Kenne
Company: kenneelectric llc, website: www.kenneelectric.com
tilte: Data scientist
date: 5 - 02 - 2020
Address
Branche 1: 10130-204 douglas oaks cir Tampa FL 33610 USA
Branche 2: 216 Rue Albert Ebe Nlonkak Yaounde-Cameroon
Tel: 0018562783591/ CM: 653180085
email:k.electric@kenneelectric.com

Disclamer

We are not SAR-COV2 0R COVID19 expert and not medical Doctor
We are Data Scientist, Data analysis, Statistician
We analyze data for decision making
We hope this webinar can help you understand Decision bases on data analysis technics

Scenario
The following scenario is about COVID19 outbreak in Cameroon as 02-24-2020.
Suppose the 10 regions of Cameroon with a population of 26390686,population density of 52, urban population of 14941523(56.3%),median age 19 yrs, experience an outbreak of the strength of COVID19 that everyone will eventually recover after (i) days and which will grants lifelong immunity once recovered.
When the pandemic was first declared by the government on 02-24-2020, one person was infected and zero person already immune to the disease.
Suppose that a susceptible person has a (p) chance of contacting a member of the infected population if contact occurs, (c) out of every(k) people will fall ills with the COVID19

What do we know about COVID-19 as of 4/11/2019

All Models are Wrong , Some are useful with uncertainty around

I- TIME DEPENDENT MODEL SIR-COVID-19-CAMEROON
Population Change Over Time

Definition and Term
We take all people in Cameroon and divide them into 3 groupes

susceptible 
𝑆(𝑡)
S(t)
All people in Cameroon Capable of becomming sick from the COVID-19
Infected 
𝐼(𝑡)
I(t)
All people in Cameroon infected by the COVID-19(when infected you get it only one time and live the susceptible population
Recorvered 
𝑅(𝑡)
R(t)
All people in Cameroon that recovered or died from COVID-19( can not be reinfected and give to others).Recovered groupe was infected and transition to recovered status to fully recovered or died
𝑆(𝑡)+𝐼(𝑡)+𝑅(𝑡)=𝑁
S(t)+I(t)+R(t)=N
(1) ( total population)
System of differential equation
we want to know the rate of change of each groupe
𝑑𝑆
𝑑𝑡

dSdt
how the number of susceptible is changing over time.Depend on probability of proximity between individual(interaction or likelyhood of interaction
𝑑𝐼
𝑑𝑡

dIdt
how the number of infection is changing over time. S transit to I to get infected
𝑑𝑅
𝑑𝑡

dRdt
how the number of recorvered is changing over time
Reproductive rate or degree of spreadability(
𝑅
𝑟
𝑜
Rro
)
Function of:
1-duration of infectivity of affected patient
2-Infectiousness of the organism
3-number of susceptible population
4-environmental condition
Values of R0 of well-known infectious diseases
COVID-19 Respiratory droplets : 3.8–8.9
5 - Bahavior of the infected population
it is how much can one person infect others (secondary infection) . it is the number of new cases that existing cases generates on average over the infection perode in the susceptible population.
it is how much can one person infect (secondary infection) others. it is the number of new cases that existing cases generates on average over the infection perode in the susceptible population.
(
𝑅
𝑟
𝑜=
𝑎
𝑆
𝑜
𝑏

=
𝑆
𝑜
𝑞)
(Rro=aSob=Soq)
𝑅
𝑒
=
𝑛𝑒𝑤𝐶𝑎𝑠𝑒𝑠
𝑒𝑥𝑖𝑠𝑡𝑖𝑛𝑔𝐶𝑎𝑠𝑒𝑠

Re=newCasesexistingCases
: effective reproductive rate
Serial intervales(
𝑆
𝐼
SI
)
is defined as the time duration between a primary case-patient (infector) having symptom onset and a secondary case-patient (infectee) having symptom onset (2). The distribution of COVID-19 serial intervals is a critical input for determining the basic reproduction number and the extent of interventions required to control an epidemic(source:https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article) reference:https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3057478/
Recovered coefficient (b) :
It is 1/number of days it takes to revover from the desease(k). 
(
1
𝑘

=𝑏)
(1k=b)
is the mean period of time during which an infected individual can pass it on.
Transmition coeficient (a):
a: describes the effective contact rate of the disease: an infected individual comes into contact with n other individuals per unit time (of which the fraction that are susceptible to contracting the disease is 
𝑆
𝑁

SN
).
𝑎=𝑝𝑞
a=pq
where: 
p: probability of becomming in contact with didease
q: chance of falling ills with the disease if you are in contact
Assumption to the Model
In mathematical modeling we have to make assumption to simplify analysis
1 - Population remain constant(the disease is short)
2 - Rate of infection proportional to the contact between the susceptible and infected at constant rate
3- infected will fully recover or died at constant rate
An idividual has following stage:
stage1: Exposed (
𝑛
𝐸
nE
: number of exposed)
An individual is infected has no symptoms and does not yet infect others, the average duration of the exposed stage is 
𝑡
𝐸
tE
stage 2: Latent Infection: (
𝑛
𝐼
nI
: number of latent infection)
An individual is infected has no symptoms, but does infect others. the average duration of latent infection is 
𝑡
𝐼
tI
(Serial intervales(
𝑆
𝐼
SI
). The individual infect Rro others individual during this period
stage 3: Isolation
After diagnosis, measure are taken to prevent infection (isolating the patient)
𝑅
𝑟
𝑜=1+𝐾(
𝑡
𝐸
+
𝑡
𝐼
)+
𝐾
2
𝑡
𝐸
𝑡
𝐼
Rro=1+K(tE+tI)+K2tEtI
(2)
where:
𝐾=
𝑑𝐿𝑛(𝐼)
𝑑𝑡

K=dLn(I)dt
Equation assume that newly infected patient may start transmitting the desease directly after getting infected¶
Limitation of 
𝑅
𝑟
𝑜
Rro
problematic if there are intermediate vector betweens groupe(malaria,typhoid)
Using COVID19-CAMEROON data,we assume that the median time from onset to clinical recovery for mild cases is approximately 2 weeks(15days) and is 3-6 weeks(42days) for patients with severe or critical disease.
