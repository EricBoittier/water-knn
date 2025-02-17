!gradients.f90
MODULE grad_module
IMPLICIT NONE


!module that calculates the gradients of a ResNet given the number of blocks and weights/biases
!use with care for nblock > 5, I have not tested it explicitly
!Efficiency wise i am not sure if this is the best way to code that

!!AT THE MOMENT the size of the descriptor (here 10) is hard coded, maybe we can find a general
!!way to code this.

!!h_in and w are initialized to the size they would have if we used nblock=7. Not sure if this
! is really appropriate
CONTAINS
    function actf_prime(i) result(j)
    implicit none
      real*8, dimension(10), intent(in) :: i ! input
      real*8, dimension(10)            :: j ! output
      j = exp(i) / (exp(i) + 1.0)
    end function

    function actf(i) result(j)
    implicit none
      real*8, dimension(10), intent(in) :: i ! input
      real*8, dimension(10)            :: j ! output
      j = log(exp(i) + 1.0 ) - log(2.0)
    end function
    
    function softplus(i, nn_width) result(j)
    implicit none
      integer, intent(in) :: nn_width ! just to allocate the correct dimension
      real*8, dimension(nn_width), intent(in)  :: i ! input
      real*8, dimension(nn_width) :: j ! output      
      j = log(exp(i) + 1.0 )
    end function
    
    function softplus_prime(i, nn_width) result(j)
    implicit none
      integer, intent(in) :: nn_width ! just to allocate the correct dimension
      real*8, dimension(nn_width), intent(in)  :: i ! input
      real*8, dimension(nn_width) :: j ! output      
      j = exp(i) / (exp(i) + 1)
    end function

    function drker33(x,xi) !function calculating the 33 kernels
    implicit none
    real*8, intent(in) :: x, xi
    real*8 :: drker33, xl, xs

    xl = x
    xs = xi
    if (x .le. xi) then
      xl = xi
      xs = x
    end if

    drker33=3d0/(20d0*xl**4) - 6d0/35d0 * xs/xl**5 + 3d0/56d0 * xs**2/xl**6

    end function drker33



    function dkdrker33(x,xi) !function calculating the derivatives of the 33 kernels
    implicit none
    real*8, intent(in) :: x, xi
    real*8 :: dkdrker33

    if (x .le. xi) then
        dkdrker33 = 3.0d0/28.0d0 * x/xi**6 - 6.0d0/(35.0d0*xi**5)
    else
        dkdrker33 = -3.0d0/(5.0d0*x**5)  + 6.0d0/7.0d0 *xi/x**6 - 9.0d0/28.0d0*xi**2/x**7
    end if

    end function dkdrker33


END MODULE


