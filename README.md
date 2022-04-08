# online-pricing
### Structure 
#### class Simulator:
- ****Members****:
  - affinity matrix for each client group, made uncertain (some distribution) in step 5
  - dictionary: first & secondary product for each product (take 2 edges with max affinity). FIXED by assumption
  - 4 possible prices for each product
  - current price of every product: arm the learner pulls
  - lambda
  - demand curve for each product, for each of 3 groups (Simplest case: a straight line + some noise; else functional data with sklearn library). **For step 7** make them change along time. **NB** uncertain conversion rates means exaclty the demand curve has noise
  - units sold discrete distribution parameters (discrete weibull makes sense!) for each group 
  - parameters of distribution of daily potential customers
  - alpha ratios, later when uncertain (**step 4**) dirichlet distribution parameters for alpha ratios for each group of people
  - total number of each group


- __Methods__:
  - **sim_one_day()**:
    - obtain number of potential customers of each product by sampling dirichlet distribution
    - for each user:
      - **sim_one_user(day, product_landed)** and store the results
    - Update the prices of the products according to the learner output, as said on **step 2**


  - **sim_one_user(day, product_landed)**:
    1. Obtain the user's reservation price sampling from the demand curve
    2. **If price is lower than user's r price**,
       1. store quantity (Sample from a discrete distribution the number of units) of units sold of that product in that day
       2. with probability taken from matrix, go to secondary product, repeat C, without further secondary products
       3. with proba lambda * p taken from matrix go to second secondary product, decide if buy
       4. From the previous steps, store all data (if clicked or not, quanityt bought)

**Important note**:     from step 3 to 6, the changes are only on the members, but the functionality of the class is the same.
For example, the demand curves on step 7 have to change over time. But we just need to add a helper function to update those members and we're done!

