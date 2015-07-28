#include "Network.h"

#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <fstream>
//#include <cstdint>
#include <string>
#include <cstring>
#include <cstdio>
#include <ctime>

using namespace std;

vector<double> inVals;
vector<double> resultVals;
vector<double> goalVals;

Network myNetwork;

int reverseInt (int i) 
{
    unsigned char x1, x2, x3, x4;

    x1 = i & 255;
    x2 = (i >> 8) & 255;
    x3 = (i >> 16) & 255;
    x4 = (i >> 24) & 255;

    return ((int)x1 << 24) + ((int)x2 << 16) + ((int)x3 << 8) + x4;
}

void train(int amount){
	
    ifstream file ("train-images-idx3-ubyte");
	ifstream file2 ("train-labels-idx1-ubyte");
	
    int magic_n=0;
    int n_of_images=0;
    int n_rows=0;
    int n_columns=0;
    file.read((char*)&magic_n,sizeof(magic_n)); 
    magic_n= reverseInt(magic_n);
    file.read((char*)&n_of_images,sizeof(n_of_images));
    n_of_images= reverseInt(n_of_images);
    file.read((char*)&n_rows,sizeof(n_rows));
    n_rows= reverseInt(n_rows);
    file.read((char*)&n_columns,sizeof(n_columns));
    n_columns= reverseInt(n_columns);
		
    int magic_n2=0;
    int n_of_images2=0;
	
	file2.read((char*)&magic_n2,sizeof(magic_n2)); 
	magic_n2= reverseInt(magic_n2);
	
    file2.read((char*)&n_of_images2,sizeof(n_of_images2));
    n_of_images2= reverseInt(n_of_images2);
	
    for(int i=0;i<amount;++i)
    {	
		inVals.clear();
		
        for(int r=0;r<28;++r)
        {
            for(int c=0;c<28;++c)
            {
                unsigned char temp=0;
				
                file.read((char*)&temp,sizeof(temp));
				
				double in = ((double)(int)temp) / 255.0;
	
				in*=2.0;
				in-=1.0;
				
				inVals.push_back(in);	
            }
        }
		
		myNetwork.FdFwd(inVals);
		
        unsigned char label=0;
        file2.read((char*)&label,sizeof(label));
		
		for(int x= 0; x<10; x++){		
			goalVals.push_back(-1.0);	
		}
		
		goalVals[(int)label] = 1.0;
		
		myNetwork.backProp(goalVals);
		
		goalVals.clear();
    }	
}

int test(){
	
    ifstream train ("t10k-images-idx3-ubyte");
	ifstream trainlabel ("t10k-labels-idx1-ubyte");
	
    int magic_n=0;
    int n_of_images=0;
    int n_rows=0;
    int n_columns=0;
    int magic_n2=0;
    int n_of_images2=0;
		
		
        train.read((char*)&magic_n,sizeof(magic_n)); 
        magic_n= reverseInt(magic_n);
        train.read((char*)&n_of_images,sizeof(n_of_images));
        n_of_images= reverseInt(n_of_images);
        train.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        train.read((char*)&n_columns,sizeof(n_columns));
        n_columns= reverseInt(n_columns);
		
		trainlabel.read((char*)&magic_n2,sizeof(magic_n2)); 
		magic_n2= reverseInt(magic_n2);
		
        trainlabel.read((char*)&n_of_images2,sizeof(n_of_images2));
        n_of_images2= reverseInt(n_of_images2);
		
		
		int error = 0;
		
		for(int i = 0; i<10000; i++){	
			inVals.clear();
            for(int r=0;r<28;++r)
            {
                for(int c=0;c<28;++c)
                {
                    unsigned char temp=0;
					
                    train.read((char*)&temp,sizeof(temp));
					
					double in = ((double)(int)temp) / 255.0;
					
					in*=2.0;
					in-=1.0;
								
					inVals.push_back(in);	
                }
            }
			
			myNetwork.FdFwd(inVals);
			myNetwork.getResults(resultVals);
			
            unsigned char label=0;
            trainlabel.read((char*)&label,sizeof(label));
			
			double maxr = resultVals[0];
			int maxindex = 0;
			for(int x= 1; x<10; x++){	
				if(resultVals[x]>maxr){
					maxr= resultVals[x];
					maxindex = x;
				}	
			}
			
			if(((int)label) != maxindex){
				error++;
			}
		}		
		return error;	
}

void testingFunc(){
	
	printf("Should be One: %f\n\n",myNetwork.m_levels[0][4].m_outputWeights[6].weight);
	myNetwork.allocmemGPU();
	double *invals = new double[784];
	inVals.clear();
	resultVals.clear();
	
	for(int i=0;i<784;++i){
		invals[i]=1.0;
		inVals.push_back(1.0);
	}
	
	myNetwork.FdFwd(inVals);
	
	myNetwork.getResults(resultVals);
	
	printf("Size %lu\n",resultVals.size());
	
	for(int i=0; i<resultVals.size(); i++){
		
		printf("%f\n", resultVals[i]);
	}

	printf("allocating GPU Network\n");
	
	printf("Feeding forward in para\n");
	myNetwork.FdFwdParallel(invals);
	
	printf("Copying Network To CPU\n");
	myNetwork.copyGpuToCpu();
	
	printf("Dealloc Network\n");
	myNetwork.deallocmemGPU();
	
	resultVals.clear();
	
	myNetwork.getResults(resultVals);
	
	printf("\n\n\n");
	
	for(int i=0; i<10; i++){
		printf("%f\n",resultVals[i]);
	}
}

void testingFunc2(){
	
	printf("Should be One: %f\n\n",myNetwork.m_levels[0][4].m_outputWeights[6].weight);
	myNetwork.allocmemGPU();
	double *invals = new double[784];
	inVals.clear();
	resultVals.clear();
	
	for(int i=0;i<784;++i){
		invals[i]=1.0;
		inVals.push_back(1.0);
	}
	
	myNetwork.FdFwd(inVals);
	myNetwork.FdFwdParallel(invals);
	
	myNetwork.getResults(resultVals);
	
	myNetwork.getResultsFromGPU();
	
	printf("CPU\n");
	for(int i=0; i<resultVals.size(); i++){
		
		printf("%f\n",resultVals[i]);
	}
	printf("GPU\n");
	for(int i=0; i<resultVals.size(); i++){	
		printf("%f\n",myNetwork.results_h[i]);
	}
	
	double *tvals = new double[10];
	for(int x= 0; x<10; x++){	
		goalVals.push_back(-1.0);
		tvals[x]=-1.0;
	}
	
	goalVals[2] = 1.0;
	tvals[2] = 1.0;
	
	printf("Backpropagating\n");
	myNetwork.backProp(goalVals);
	myNetwork.backPropParallel(tvals);
	
	resultVals.clear();
	
	myNetwork.FdFwd(inVals);
	myNetwork.FdFwdParallel(invals);
	
	myNetwork.getResults(resultVals);
	
	myNetwork.getResultsFromGPU();
	
	printf("CPU\n");
	for(int i=0; i<resultVals.size(); i++){	
		printf("%f\n",resultVals[i]);
	}

	printf("GPU\n");
	for(int i=0; i<resultVals.size(); i++){	
		printf("%f\n",myNetwork.results_h[i]);
	}
	
	myNetwork.deallocmemGPU();
}

void train_feed_on_gpu(int amount){	

    ifstream file ("train-images-idx3-ubyte");
	ifstream file2 ("train-labels-idx1-ubyte");	
	
    int magic_n=0;
    int n_of_images=0;
    int n_rows=0;
    int n_columns=0;
    file.read((char*)&magic_n,sizeof(magic_n)); 
    magic_n= reverseInt(magic_n);
    file.read((char*)&n_of_images,sizeof(n_of_images));
    n_of_images= reverseInt(n_of_images);
    file.read((char*)&n_rows,sizeof(n_rows));
    n_rows= reverseInt(n_rows);
    file.read((char*)&n_columns,sizeof(n_columns));
    n_columns= reverseInt(n_columns);
		
    int magic_n2=0;
    int n_of_images2=0;
	
	file2.read((char*)&magic_n2,sizeof(magic_n2)); 
	magic_n2= reverseInt(magic_n2);
	
    file2.read((char*)&n_of_images2,sizeof(n_of_images2));
    n_of_images2= reverseInt(n_of_images2);
	
	double *invals = new double[784];
	
    for(int i=0;i<amount;++i)
    {		
		for(int x=0;x<784;x++)
		{
			invals[x]=0.0;
		}		
        for(int r=0;r<28;++r)
        {
            for(int c=0;c<28;++c)
            {
                unsigned char temp=0;
				
                file.read((char*)&temp,sizeof(temp));
				double in = ((double)(int)temp) / 255.0;
				
				in*=2.0;
				in-=1.0;

				invals[(28*r)+c] = in;	
            }
        }
		myNetwork.allocmemGPU();	

		myNetwork.FdFwdParallel(invals);
		
		myNetwork.copyGpuToCpu();
		
		myNetwork.deallocmemGPU();
		
        unsigned char label=0;
        file2.read((char*)&label,sizeof(label));
		
		for(int x= 0; x<10; x++){	
			goalVals.push_back(-1.0);	
		}
		
		goalVals[(int)label] = 1.0;
		
		myNetwork.backProp(goalVals);
		
		goalVals.clear();
    }	
}

double *tvals = new double[10];

void train_on_gpu(int amount){	

    ifstream file ("train-images-idx3-ubyte");
	ifstream file2 ("train-labels-idx1-ubyte");	
	
    int magic_n=0;
    int n_of_images=0;
    int n_rows=0;
    int n_columns=0;
    file.read((char*)&magic_n,sizeof(magic_n)); 
    magic_n= reverseInt(magic_n);
    file.read((char*)&n_of_images,sizeof(n_of_images));
    n_of_images= reverseInt(n_of_images);
    file.read((char*)&n_rows,sizeof(n_rows));
    n_rows= reverseInt(n_rows);
    file.read((char*)&n_columns,sizeof(n_columns));
    n_columns= reverseInt(n_columns);	
	
    int magic_n2=0;
    int n_of_images2=0;
	
	file2.read((char*)&magic_n2,sizeof(magic_n2)); 
	magic_n2= reverseInt(magic_n2);
	
    file2.read((char*)&n_of_images2,sizeof(n_of_images2));
    n_of_images2= reverseInt(n_of_images2);

	double *invals = new double[784];
	
    for(int i=0;i<amount;++i)
    {
		for(int x=0;x<784;x++)
		{
			invals[x]=0.0;
		}

        for(int r=0;r<28;++r)
        {
            for(int c=0;c<28;++c)
            {
                unsigned char temp=0;
				
                file.read((char*)&temp,sizeof(temp));
				
				double in = ((double)(int)temp) / 255.0;
	
				in*=2.0;
				in-=1.0;

				invals[(28*r)+c] = in;
            }
        }
		
		myNetwork.FdFwdParallel(invals);

        unsigned char label=0;
        file2.read((char*)&label,sizeof(label));
		
		for(int x= 0; x<10; x++){

			tvals[x]=-1.0;
		}
	
		tvals[(int)label] = 1.0;
		
		myNetwork.backPropParallel(tvals);
    }	
}

int test_on_gpu(){
	
    ifstream train ("t10k-images-idx3-ubyte");
	ifstream trainlabel ("t10k-labels-idx1-ubyte");
	
    int magic_n=0;
    int n_of_images=0;
    int n_rows=0;
    int n_columns=0;
    int magic_n2=0;
    int n_of_images2=0;
	int error = 0;	
		
        train.read((char*)&magic_n,sizeof(magic_n)); 
        magic_n= reverseInt(magic_n);
        train.read((char*)&n_of_images,sizeof(n_of_images));
        n_of_images= reverseInt(n_of_images);
        train.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        train.read((char*)&n_columns,sizeof(n_columns));
        n_columns= reverseInt(n_columns);
		
		trainlabel.read((char*)&magic_n2,sizeof(magic_n2)); 
		magic_n2= reverseInt(magic_n2);
		
        trainlabel.read((char*)&n_of_images2,sizeof(n_of_images2));
        n_of_images2= reverseInt(n_of_images2);
				
		double *invals = new double[784];

	    for(int i=0;i<10000;++i)
	    {		
			for(int x=0;x<784;x++)
			{
				invals[x]=0.0;
			}
		
	        for(int r=0;r<28;++r)
	        {
	            for(int c=0;c<28;++c)
	            {
	                unsigned char temp=0;	
				
	                train.read((char*)&temp,sizeof(temp));
				
					double in = ((double)(int)temp) / 255.0;
					
					in*=2.0;
					in-=1.0;
				
					invals[(28*r)+c] = in;
	            }
	        }
			
			myNetwork.FdFwdParallel(invals);
			myNetwork.getResultsFromGPU();
			
            unsigned char label=0;
            trainlabel.read((char*)&label,sizeof(label));
			
			//max result
			double maxr = myNetwork.results_h[0];
			int maxindex = 0;
			for(int x= 1; x<10; x++){
				
				if(myNetwork.results_h[x]>maxr){
					maxr= myNetwork.results_h[x];
					maxindex = x;
				}
				
			}
			
			if(((int)label) != maxindex){
				error++;
			}
		}
		return error;
}


int main(int argc, char** argv)
{
	srand (time(NULL));
	vector<unsigned> topol;
	
	topol.push_back(784);
	topol.push_back(2500);
	topol.push_back(500);
	topol.push_back(49);
	topol.push_back(10);
	
	myNetwork.init(topol);

	myNetwork.allocmemGPU();
	
	//train FdFwd then back propagation
	
	double error;
	
		train_on_gpu(60000);
		error = ((double)test_on_gpu())/ 10000.0;
		cout<<"The Error is: " << error << endl;
		myNetwork.copyGpuToCpu();
		myNetwork.outputToFile("Networks/784-2000-500-49-10_0");

	myNetwork.deallocmemGPU();
	
	cout<<"DONE"<<endl;	
}


