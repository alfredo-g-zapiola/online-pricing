## Formulation of the optimisation problem


### The problem formulation is:


<img src="https://latex.codecogs.com/svg.image?&space;&space;&space;&space;\max_{\underline{x}&space;\in&space;\mathcal{P}}&space;\sum_{g=1}^{3}&space;\sum_{p&space;\in&space;P}\alpha_{gp}&space;(c_{gp}(x_p)*m_p(x_p)&space;&plus;&space;\sum_{\tilde{p}&space;\in&space;P&space;:&space;\tilde{p}&space;\neq&space;p&space;}&space;\tilde{i}_{p\tilde{p}}(x_{\tilde{p}})&space;*&space;c_{g\tilde{p}}(x_p)*&space;m_{\tilde{p}}(x_{\tilde{p}})&space;)&space;">    

where:
- __x__ is the vector of the price for each product
- g denotes the group
- P = {1, 2, ..., 5} denotes the set of product indices
- c_gp indicates the conversion rate of group g for product p at price x_p
- m_p(x_p) is the margin for product p given the its price x_p
- i_pptilde is the influence probability given the first visited webpage was p for product ptilde, which depends on x_p

###### useful [link] for the render of the equation

[link]: https://latex.codecogs.com/