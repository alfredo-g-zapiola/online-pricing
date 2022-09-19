##
##In this file we obtain the functional distributions for the demand curves
# The idea is to use Gaussian Processes with exponential covariance function.
# cf. https://github.com/AndreaCappozzo/lab_nonparametricstatistics/blob/main/Block%20I%20-%20Nonparametric%20data%20exploration/NPS-lab02_funct_depth_measures.html


# Some functions we will use
library(roahd)
# function to control rate is between 0 and 1.
wrapper.f <- function(f.t){
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

plot.product <- function(f, min.price, max.price, n = 10){
  P <- 101  
  grid <-  seq( min.price, max.price, length.out =  P)
  alpha <-  0.05
  beta <-  0.0025
  C_st <- exp_cov_function( grid, alpha, beta )
  m <- (sapply(grid, f))
  data <- generate_gauss_fdata(N = n,centerline = m,Cov=C_st)
  matplot(grid,t(data), type="l", col=adjustcolor(col=1,alpha.f = .4))
  lines(grid,m, col="blue", lwd=5)
  
}
sample.demand = function(f,p, min.price, max.price){
  P <- 201
  grid <-  seq( min.price, max.price, length.out =  P)
  alpha <-  0.05
  beta <-  0.0025
  C_st <- exp_cov_function( grid, alpha, beta )
  m <- (sapply(grid, f))
  data <- generate_gauss_fdata(N = 1,centerline = m,Cov=C_st)
  sapply(data, wrapper.f)
  data = make.monotone(data)
  #matplot(grid,t(data), type="l", col=adjustcolor(col=1,alpha.f = .4))
  #lines(grid,m, col="blue", lwd=5)
  i = which(grid == p)
  return (data[i])
}



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
# margins calculation:
# we assume 40% margin so that we never lose money

# P1: ECHO dot
#
set.seed(23)
cr.echo <- sort(runif(4, 0, .5))
cr.echo[2] =cr.echo[4] * 1.2 
plot(df$echo_dot, c(cr.echo,.9), ylim = c(0,1), xlim=c(0,40))
a <- lm(c(cr.echo,.9) ~  + I(df$echo_dot))
lines(df$echo_dot, predict(a, I(df$echo_dot)))
#The true function is thus
echo_dot <- function(p){0.83900346 - 0.01589725 *p}
lines(df$echo_dot, sapply(df$echo_dot, echo_dot_rich  ))

echo_dot_rich <- function(p){ 1 -  0.01589725*.8 *p}
lines(df$echo_dot, sapply(df$echo_dot, echo_dot_poor  ))
echo_dot_poor <- function(p){0.83900346*.8 - 0.01589725*1.2 *p}
echo_dot_poor(34)

echo_dot <- function(p){0.83900346 - 0.01589725 *p}
echo_dot_rich <- function(p){ 1 -  0.01589725*.8 *p}
echo_dot_poor <- function(p){0.83900346*.8 - 0.01589725*1.2 *p}

## P2: Ring_chime
set.seed(98)

cr.ring <- sort(c(runif(4, 0, .5), runif(1, .7, 1)))
cr.ring

plot(df$ring_chime, cr.ring, ylim = c(0,1), xlim=c(0,40))
a <- lm(cr.ring ~  + I(df$ring_chime))
lines(df$ring_chime, predict(a, I(df$ring_chime)))
lines(df$ring_chime, sapply(df$ring_chime, ring_chime_rich  ))
lines(df$ring_chime, sapply(df$ring_chime, ring_chime_poor  ))

ring_chime <- function(p){0.85502936 -0.02193963*p}
ring_chime_rich <- function(p){1 -0.02193963*.7*p}
ring_chime_poor <- function(p){0.85502936*.7 -0.02193963*1.3*p}


## P3 ring f
set.seed(20011999)

cr.ring.f <- sort(c(runif(4, 0, .5), runif(1, .7, 1)))
cr.ring.f

plot(df$ring_f, cr.ring.f, ylim = c(0,1), xlim=c(0,200))
a <- lm(cr.ring.f ~  1 + I(df$ring_f))
lines(df$ring_f, predict(a, I(df$ring_f)))
lines(df$ring_f, sapply(df$ring_f, ring_f_rich))
lines(df$ring_f, sapply(df$ring_f, ring_f_poor))

ring_chime_f <- function(p){0.819312461  - 0.003097635 *p}
ring_chime_f_rich <- function(p){1 -0.003097635*.7*p}
ring_chime_f_poor <- function(p){0.819312461*.7 -0.003097635*1.3*p}

##  P4 ring video
set.seed(22200337)

cr.ring.v <- sort(c(runif(4, 0, .5), runif(1, .7, 1)))
cr.ring.v

plot(df$ring_v, cr.ring.v, ylim = c(0,1), xlim=c(0,60))
a <- lm(cr.ring.v ~  1 + I(df$ring_v))
lines(df$ring_v, predict(a, I(df$ring_v)))
lines(df$ring_v , sapply(df$ring_v, ring_v_rich))
lines(df$ring_v, sapply(df$ring_v, ring_v_poor))

ring_v <- function(p){0.722883157  - 0.006319785 *p}
ring_v_rich <- function(p){1 -0.006319785*.7*p}
ring_v_poor <- function(p){0.722883157*.7 -0.006319785*1.3*p}


# P5 echo_show
## ring f
set.seed(21071065)

cr.echo.show <- sort(c(runif(4, 0, .5), runif(1, .7, 1)))
cr.echo.show

plot(df$echo_show, cr.echo.show, ylim = c(0,1), xlim=c(0,96))
a <- lm(cr.echo.show ~  1 + I(df$echo_show))
lines(df$echo_show, predict(a, I(df$echo_show)))
lines(df$echo_show, sapply(df$echo_show, echo_show_rich))
lines(df$echo_show, sapply(df$echo_show, echo_show_poor))

echo_show <- function(p){0.885669046  - 0.007681115 *p}
echo_show_rich <- function(p){1 -0.007681115*.7*p}
echo_show_poor <- function(p){0.885669046*.7 -0.007681115*1.3*p}
