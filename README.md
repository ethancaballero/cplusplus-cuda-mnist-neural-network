C++_Cuda_Neural_Network_MNIST_train_test
===================

Basic Neural Network implemented in parallel on the graphics card using cuda and c++. Trains and tests on digits from the minst database.


#run these in CLI to configure nnvc CUDA compiler driver after installation:
export PATH=/Developer/NVIDIA/CUDA-7.0/bin:$PATH

export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-7.0/lib:$DYLD_LIBRARY_PATH

kextstat | grep -i cuda

nvcc -V
