
library(roahd)
set.seed(20)

# a function to control the sampled conversion rate is between 0 and 1
clipper.f <- function(f.t){
      if (f.t > 1){
        return (1)
      }
      else if (f.t < 0){
        return (0)
      }
      else{
        return (f.t)
      }

}
# function to ensure monotonicity
make.monotone <- function(data){
      k = 2
      while(k <= length(data)){
        if (data[k] > data[k - 1]){
          data[k] = data[k - 1]
        }
        k = k + 1
      }
      return (data)
}
# Given a function (i.e the mean of a functional distribution), and a price,
# obtain a sample and return its value at price p
sample.demand = function(f,p, min.price, max.price){
  P <- 201
  grid <-  seq( min.price, max.price, length.out =  P)
  alpha <-  0.05
  beta <-  0.0025
  C_st <- exp_cov_function( grid, alpha, beta )
  m <- (sapply(grid, f))
  data <- generate_gauss_fdata(N = 1,centerline = m,Cov=C_st)
  data = sapply(data, clipper.f)
  data = make.monotone(data)
  # to plot and visualise
  #matplot(grid,t(data), type="l", col=adjustcolor(col=1,alpha.f = .4))
  #lines(grid,m, col="blue", lwd=5)
  i = which(grid == as.integer(p))
  return (data[i])
}

# the means for each product
# P1: echo dot
echo_dot <- function(p){0.83900346 - 0.01589725 *p}
echo_dot_rich <- function(p){ 1 -  0.01589725*.8 *p}
echo_dot_poor <- function(p){0.83900346*.8 - 0.01589725*1.2 *p}

# P2 Ring chime
ring_chime <- function(p){0.85502936 -0.02193963*p}
ring_chime_rich <- function(p){1 -0.02193963*.7*p}
ring_chime_poor <- function(p){0.85502936*.7 -0.02193963*1.3*p}

# P3 Ring Floodlight Cam
ring_f <- function(p){0.819312461  - 0.003097635 *p}
ring_f_rich <- function(p){1 -0.003097635*.7*p}
ring_f_poor <- function(p){0.819312461*.7 -0.003097635*1.3*p}

# P4 Ring video doorbell
ring_v <- function(p){0.722883157  - 0.006319785 *p}
ring_v_rich <- function(p){1 -0.006319785*.7*p}
ring_v_poor <- function(p){0.722883157*.7 -0.006319785*1.3*p}

# P5 echo show
echo_show <- function(p){0.885669046  - 0.007681115 *p}
echo_show_rich <- function(p){1 -0.007681115*.7*p}
echo_show_poor <- function(p){0.885669046*.7 -0.007681115*1.3*p}

# build dataframe for products' prices
prezzi = c(1, .95, .8, .4, 0)
prodotti = c("echo_dot", "ring_chime", "ring_f",
             "ring_v", "echo_show")
prezzi_init = c(34, 36, 200, 60, 96) # the prices on amazon
df = data.frame(matrix(0, 4, 5))
names(df) = prodotti
for (p in 1:5){
  for (pi in 1:5){
    df[p, pi] = prezzi_init[pi] * prezzi[p]
  }

}


print("R initialisation complete and successfull")
