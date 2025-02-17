
program NNrWater
    use grad_module
    implicit none
    

    !Here we initialize all the required weights and biases according to
    !the input to the NN, its architecture, etc.
    integer :: n_input, n_hidden, i, ii, jj, nn_width, natom, id
    real*8 :: E, b_out, meanE, stdE
    real*8, dimension(:,:,:), allocatable :: w_hidden !weights of the hidden layers
    real*8, dimension(:,:), allocatable :: w_in, b_hidden
    real*8, dimension(:), allocatable :: h1_in, h1_out, h2_in, h2_out, h3_in, h3_out, w_out
    real*8, dimension(:,:), allocatable :: pos0, forces
    real*8, dimension(3) :: r0, dEdr
    character(len=6)::symbol
    natom = 3
    n_input = 3
    n_hidden = 2
    nn_width = 20
    
    allocate(w_in(n_input,nn_width))
    allocate(h1_in(nn_width))
    allocate(h1_out(nn_width))
    allocate(h2_in(nn_width))
    allocate(h2_out(nn_width))
    allocate(h3_in(nn_width))
    allocate(h3_out(nn_width))
    allocate(w_out(nn_width))
    allocate(b_hidden((n_hidden+1), nn_width))
    allocate(w_hidden(nn_width, nn_width, n_hidden))
    allocate(pos0(natom, 3))
    allocate(forces(natom, 3))
    
    open(unit=10,file="inp.xyz", status="old")
    read(10,*) 
    read(10,*)
    do ii = 1, natom 
        read(10,*) symbol, pos0(ii,:)
    end do


    !calculate interatomic distances

    r0(1) = sqrt((pos0(1,1)-pos0(2,1))**2 + (pos0(1,2)-pos0(2,2))**2 + (pos0(1,3)-pos0(2,3))**2)
    r0(2) = sqrt((pos0(2,1)-pos0(3,1))**2 + (pos0(2,2)-pos0(3,2))**2 + (pos0(2,3)-pos0(3,3))**2)
    r0(3) = sqrt((pos0(1,1)-pos0(3,1))**2 + (pos0(1,2)-pos0(3,2))**2 + (pos0(1,3)-pos0(3,3))**2)     

    open(11, file="wandb/w_in") !corresponds to the weight of first layer
    read(11,*) w_in
    !write(*,*) w_in
    
    open(12, file="wandb/b_hidden") !corresponds to the biases of all but the output layer
    do jj = 1, (n_hidden+1)
        do ii = 1,nn_width
            read(12,*) b_hidden(jj, ii)
        end do
    end do
    
    open(13, file="wandb/w_hidden")
    !read weights (total of nhidden blocks)
    read(13,*) w_hidden
    !write(*,*) shape(w_hidden)

    open(14, file="wandb/w_out")
    !read weights of the output layer
    read(14,*) w_out
    !write(*,*) w_out

    
    open(15, file="wandb/b_out") !corresponds to the bias of the last dense layer with no activation
    read(15,*) b_out
    !write(*,*) b_out
    
    open(16, file="wandb/meanE") 
    read(16,*) meanE
    !write(*,*) meanE
    open(17, file="wandb/stdE")
    read(17,*) stdE
    !write(*,*) stdE    
    

    !Input layer
    h1_in = matmul(r0, w_in) + b_hidden(1,:)
    h1_out = softplus(h1_in, nn_width)
    
    !Hidden layers
    h2_in = matmul(h1_out,w_hidden(:,:,1))  + b_hidden(2,:)
    h2_out = softplus(h2_in, nn_width)
    
    h3_in = matmul(h2_out,w_hidden(:,:,2))  + b_hidden(3,:)
    h3_out = softplus(h3_in, nn_width)
    
    !Output layer
    E = (dot_product(h3_out, w_out) + b_out)* stdE + meanE
    !write(*,*) (dot_product(h3_out, w_out) + b_out)* stdE + meanE
    !write(*,*) E
    
    !DO THE BACK PASS HERE, i.e. dE / dr
    !write(*,*) (w_out * softplus_prime(h3_in, nn_width))
    !write(*,*) matmul((w_out * softplus_prime(h3_in, nn_width)), Transpose(w_hidden(:, :, 2))) ! THIS WORKS, and is the way I would write it.
    !This adds an additional transpose, which probably slows the evaluation... THis is why I use the below code.
    dEdr =  -matmul(w_in, matmul(w_hidden(:, :, 1), matmul(w_hidden(:, :, 2), (w_out * softplus_prime(h3_in, nn_width))) &
    * softplus_prime(h2_in, nn_width)) * softplus_prime(h1_in, nn_width)) * stdE! THIS WORKS, too.
    !write(*,*) dEdr
    
    
    
    !Do the transformation here, i.e. dE / dpos = dE/dr * dr/dpos
    forces(:,1) = (pos0(1, :) - pos0(2, :))/r0(1) * dEdr(1) + (pos0(1, :) - pos0(3, :))/r0(3) * dEdr(3)
    forces(:,2) = (pos0(2, :) - pos0(1, :))/r0(1) * dEdr(1) + (pos0(2, :) - pos0(3, :))/r0(2) * dEdr(2)
    forces(:,3) = (pos0(3, :) - pos0(1, :))/r0(3) * dEdr(3) + (pos0(3, :) - pos0(2, :))/r0(2) * dEdr(2)
    !write(*,*) forces(:, 1)
    
    
    open(unit=31,file="ener.out")
    open(unit=32,file="grad.out")

    write(31,*)"FINAL SINGLE POINT ENERGY =", E
    write(32,*)"# The current gradient"
    do ii = 1, natom
      write(32,*)forces(:,ii)
    end do
    write(32,*)"# The end"


end program NNrWater
