C++_Cuda_Neural_Network_MNIST_train_test
===================

Neural Network implemented in parallel on gpu using cuda and c++. Trains and tests on digits from the minst dataset.

#run these in CLI to configure nnvc CUDA compiler driver after cuda installation:
```
export PATH=/Developer/NVIDIA/CUDA-6.5/bin:$PATH
export DYLD_LIBRARY_PATH=/Developer/NVIDIA/CUDA-6.5/lib:$DYLD_LIBRARY_PATH
kextstat | grep -i cuda
nvcc -V
```

#to compile & run:
```
make
./Network
```
