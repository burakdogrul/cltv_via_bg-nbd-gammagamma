# **Predicting Customer Life Time Value (CLTV) via Beta Geometric / Negative Binominal Distribution (BG/NBD) and Gamma Gamma Model**

## 1-INTRODUCTION

![clv](images/clv.jpg)

**CUSTOMER LIFETIME VALUE**

CLTV is a measurement of how valuable a customer is to your company, not just on a purchase-by-purchase basis but across the whole relationship. Probabilistic lifetime value estimation is made with time projection for a certain t time. CLTV is a dynamic concept, not a static model.

The most basic formula we use is as follows:

> **CLTV = Expected Number of Transaction \* Expected Average Profit**

We will estimate the “Expected Number of Transaction” part using the BG/NBD model and the “Expexed Average Profit” part using the gamma gamma model.

Before moving on to the explanation of the models, I would like to explain the concept of buy till you die. We actually create the BG/NBD model using the concept of Buy till you die.

**BUY TILL YOU DIE MODEL**

Buy Till You Die model fit probabilistic models to historical transactional data to calculate customer lifetime value. BYTD model answers these kind of questions:

1. How many customers are active?
2. How many customers will be active one year from now?
3. Which customers have churned?
4. How valuable will any customer be to the company in the future?

**Transaction Process (Buy)**

As long as it is alive, the number of transactions to be performed by a client in a given time period is poisson distributed with the transaction rate parameter. In other words, as long as a customer is alive, he or she will continue to make random purchases around her own transaction rate. Transaction rates vary for each client and are gamma distributed for the entire population. (r,alpha)

**Dropout process (Till You Die)**

Each customer has a dropout rate (dropout probability) with probability p. A customer will churn with a certain probability after making a purchase. Dropout rates vary for each client and are beta distributed for the entire population. (a,b)

### HISTORY

![timeline](images/timeline.png)

- NBD (Ehrenberg 1959)
- Pareto/NBD Schmittlein, Morrison, and Colombo 1987)
- BG/NBD (P. Fader, Hardie, and Lee 2005)
- Pareto/NBD (HB) Ma and Liu (2007)
- MBG/NBD Batislam, Denizel, and Filiztekin (2007), Hoppe and Wagner (2007)
- Pareto/NBD (Abe) Abe (2009)
- BG/BB (Fader, Hardie, and Shang 2010)
- Pareto/GGG Platzer and Reutterer (2016)

The original NBD model from 1959 functions as a benchmark for later models because it’s based on a heterogenous purchase process. But, NBD doesn’t account for customer churn.

The next model, Pareto/NBD from 1987, adds a heterogeneous dropout process and is considered one of the top buy-til-you-die models.

Next up, the BG/NBD model adjusts assumptions to reduce computation time and offers a more robust parameter search. However, this model assumes customers without repeat transactions have not churned.

MBG/NBD removes inconsistencies with the former model by allowing customers without any activity to remain inactive.

The newer BG/CNBD-k and MBG/CNBD-k models improve forecasting accuracy by allowing for regularity in transaction times. If this regularity exists, these new models can result in much more accurate customer-level predictions.

The variants of Pareto/NBD models by Ma and Liu (2007) and by Abe (2009) utilize MCMC simulation to allow for more flexible assumptions. The first model, Pareto/NBD (HB) is a hierarchical Bayes variant that tests out this approach while sticking to the original model’s assumptions. The second variation, Pareto/NBD (Abe), can incorporate covariates.

Pareto/GGG is a third variation of Pareto/NBD that accounts for some level of regularity for inter-transaction times.

## 2-Beta Geometric / Negative Binominal Distribution (BG/NBD) MODEL

Fader, Hardie and Lee, they present the BG/NBD model as an alternative to the Pareto/NBD. The positioning of work is that the model yields very similar results to the Pareto/NBD while being vastly easier to implement.

1. While active, transactions made by a customer in time period t is Poisson distributed with mean λt
2. Differences in transaction rate between customers follows a gamma distribution with shape r and scale α
3. Each customer becomes inactive after each transaction with probability p
4. Differences in p follows a beta distribution with shape parameters a and b

It should be so that we capture the buying pattern of the whole population and then find a pattern that will personalize the buying of this whole population..

In summary, what it will do is learn the mass behavior from these individual behaviors and then make a probabilistic estimation specific to the individual.

The following should not be forgotten: After making a purchase, the customer becomes partial churn.

The BG/NBD Model probabilistically models two processes for the expected number of transactions.

*First Process:* Transaction Process (**Buy**)

*Second Process:* Dropout process (**Till You Die**) --> process of becoming churn

Formula

![bgnbdformula](images/bgnbdformula.png)



- x --> frequency of customers who have made at least two purchases
- tx --> customer's recency value (must be calculated individually for each customer)
- T --> Time since the customer's first purchase. Age of customer for company. Tenure.
- r, alfa --> difference in transaction rate between customers parameters of gamma distribution
- a,b --> Beta distribution parameters expressing drop rate

In other words, x, tx and T are the characteristics of individuals.

As a result, it will give the expected values of purchase values in a certain t period while taking values specific to individuals and carrying the characteristics of the population.

In the light of the gamma and beta distribution that we will learn from the population, the characteristics of the individual and the expected y value in a certain t period will be estimated.

2F1 is the Gaussian hypergeometric function

r,alpha,a,b are estimated using the maximum likelihood method



*NOTE-1*

**Gamma distribution**

The gamma distribution is a two-parameter family of continuous probability distributions. The exponential distribution, Erlang distribution, and chi-square distribution are special cases of the gamma distribution. There are two different parameterizations in common use:

• With a shape parameter k and a scale parameter θ.

• With a shape parameter α = k and an inverse scale parameter β = 1/θ, called a rate parameter.

In each of these forms, both parameters are positive real numbers.

The gamma distribution is the maximum entropy probability distribution



![gamma distribution](images/gamma%20distribution.png)



*NOTE-2*

**Beta distribition**

The beta distribution is a family of continuous probability distributions defined on the interval [0, 1] parameterized by two positive shape parameters, denoted by α and β, that appear as exponents of the random variable and control the shape of the distribution.

The beta distribution has been applied to model the behavior of random variables limited to intervals of finite length in a wide variety of disciplines.



## 3-Gamma Gamma Model

The monetary value of a customer’s given transaction varies randomly around their average transaction value.

Average transaction values vary across customers but do not vary over time for any given individual.

The distribution of average transaction values across customers is independent of the transaction process.

The average transaction value is gamma distributed among all customers.

Formula:

![gammagammasubmodel](images/gammagammasubmodel.png)

- mx and x parameters come from user
- x--> Frequency. The number of recurring sales (transactions made at least 2 times)
- mx --> Monetary. Observed transaction value.
- The p,q and y are parameters from distribution
- With these parameters, the expected monetary value will be estimated.

## 4-CLTV BG/NBD&GAMMA GAMMA Implementation

Data Set : https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

**Attribute Information:**

**InvoiceNo:** *Invoice number. Nominal. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'c', it indicates a cancellation.*

**StockCode:** *Product (item) code. Nominal. A 5-digit integral number uniquely assigned to each distinct product.*

**Description:** *Product (item) name. Nominal.*

**Quantity:** *The quantities of each product (item) per transaction. Numeric.*

**InvoiceDate:** *Invice date and time. Numeric. The day and time when a transaction was generated.*

**UnitPrice:** *Unit price. Numeric. Product price per unit in sterling .*

**CustomerID:** Customer number. Nominal. A 5-digit integral number uniquely assigned to each customer.

**Country:** *Country name. Nominal. The name of the country where a customer resides.*



## 5-BG/NBD Model Validation

We need to devote our model with two different sets of cross-validation method. We will then use the holdout dataset as production data, fit a new BG/NBD model on calibration data, and compare the predicted and actual number of repeat purchases over the holdout period. In my next article, I will give more detailed information about how the validation process will be.

## 6-INCREASE CUSTOMER LIFETIME VALUE

**Provide 24/7 Support**

The customer always wants to reach the company instantly. For this reason, an infrastructure should be established to serve the customer 24 hours a day, 7 days a week. The customer's problem should be resolved as soon as possible and the customer should be satisfied.

**Monitor Social Media**

It is very important to reach the customer through social media in these days when the internet has become widespread. New products and campaigns should be promoted, and if necessary, the customer with a problem should be dealt with individually. Bearing in mind the risks, your team must have at least one employee focused on tracking and replying to social media comments.

**Launch a loyalty program**

Retaining a customer is easier and less costly than acquiring a new customer. If there is not, a customer loyalty card should be created. This card can be physical or online. With this card, the customer should be encouraged to collect stars or similar items and encourage more shopping. Discounts can be defined on the customer's birthdays or special days.

**Use Up-Sells and Cross-Sells**

It may seem like a traditional method, but it still works. More monetary should be provided than the customer by encouraging the customer with up-sell or cross-selling. While doing this, the customer should not feel deceived.

**Monitor the feedback**

Feedback from the customer should not be ignored. Small problems can lead to bigger problems. Bad information spreads faster than good information on the internet. Nobody wants the company name to be badly mentioned. For this reason, the problem should be solved as soon as possible.

## 7-CONCLUSION

With the widespread use of the internet and gig economies, competition is now at its highest level. In this direction, it is very important to understand the customer. It is a known fact that it is more costly for companies to reach a new customer than to retain an existing customer. For this reason, it is very important to know which customer will make how much profit and how often they will shop. It is necessary to calculate these values correctly and to reach the customer who will be churn correctly. It is known that there are many computational methods. In this article, I wanted to show how to make a calculation using BG/NBD and gamma gamma models. Reaching the customer at the right time is as important as calculating correctly. These methods differ for companies. Companies should use and implement their own methods. Thank you for reading and taking the time to read my article.

## References

https://www.veribilimiokulu.com/

https://retina.ai/academy/lesson/history-of-buy-til-you-die-btyd-models/

Peter S.Fader, Bruce G.S. Hardie, Ka Lok Lee. December 2008

https://en.wikipedia.org/wiki/Gamma_distribution

https://en.wikipedia.org/wiki/Beta_distribution

Peter S.Fader, Bruce G.S. Hardie. February 2013

https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

https://towardsdatascience.com/what-is-your-customers-worth-over-their-lifetime-dfae277fd166
