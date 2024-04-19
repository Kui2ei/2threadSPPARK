#include <stdio.h>
#include <iostream>
using namespace std;
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <cmath>
#include <inttypes.h>
#include <gmp.h>
#include "./bigint.cuh"
#define SIZE 8
float tt=0;

__device__ uint32_t mod[SIZE] = { 
    0xf0000001, 0x43e1f593, 
    0x79b97091, 0x2833e848, 
    0x8181585d, 0xb85045b6, 
    0xe131a029, 0x30644e72};

__global__ void init(){
	M0=computeM0(mod[0]);
}

template<uint32_t limbs>
__global__ void mul_kernel(uint32_t* d_a, uint32_t* d_b, uint32_t* d_c){
	register uint32_t aa[8];
	register uint32_t bb[8];
	register uint32_t cc[8];
	copyArrayValues(aa,d_a);
	copyArrayValues(bb,d_b);
	#pragma unroll
	for(int i=0;i<4096;i++)
    	operator_mul<limbs>(aa,bb,mod,cc,SIZE*32);
	copyArrayValues(d_c,cc);
}

template<uint32_t limbs>
__global__ void add_kernel(uint32_t* d_a, uint32_t* d_b, uint32_t* d_c){
	register uint32_t aa[limbs+1];
	register uint32_t bb[limbs+1];
	register uint32_t cc[8];
	copyArrayValues(aa,d_a);
	copyArrayValues(bb,d_b);
	#pragma unroll
	for(int i=0;i<4096;i++)
		operator_add<limbs>(aa,bb,mod,cc);
	copyArrayValues(d_c,cc);
}

template<uint32_t limbs>
__global__ void sub_kernel(uint32_t* d_a, uint32_t* d_b, uint32_t* d_c){
	register uint32_t aa[limbs+1];
	register uint32_t bb[limbs+1];
	register uint32_t cc[8];
	copyArrayValues(aa,d_a);
	copyArrayValues(bb,d_b);
	#pragma unroll
	for(int i=0;i<4096;i++)
		operator_sub<limbs>(aa,bb,mod,cc);
	copyArrayValues(d_c,cc);
}

void to_mont(uint32_t* data, mpz_t mp_bigint, mpz_t MOD){
	uint32_t temp[SIZE+1]={};
	temp[SIZE]=0x00000001;

	mpz_t mpz_temp;
	mpz_init(mpz_temp);
	mpz_import(mpz_temp, SIZE+1, -1, sizeof(uint32_t), 0, 0, temp);

	mpz_mul(mp_bigint,mp_bigint,mpz_temp);
	mpz_mod(mp_bigint,mp_bigint,MOD);
	size_t mark;
	mpz_export(data, &mark, -1, sizeof(uint32_t), 0, 0, mp_bigint);
	while (mark<SIZE)
		data[mark++] = 0;

}

void mont_back(uint32_t* data, mpz_t mp_bigint, mpz_t MOD){
	uint32_t temp[SIZE+1]={};
	temp[SIZE]=0x00000001;

	mpz_t mpz_temp;
	mpz_init(mpz_temp);
	mpz_import(mpz_temp, SIZE+1, -1, sizeof(uint32_t), 0, 0, temp);
	
	mpz_invert(mpz_temp,mpz_temp,MOD);
	mpz_mul(mp_bigint,mp_bigint,mpz_temp);
	mpz_mod(mp_bigint,mp_bigint,MOD);
	size_t mark;
	mpz_export(data, &mark, -1, sizeof(uint32_t), 0, 0, mp_bigint);
	while (mark<SIZE)
		data[mark++] = 0;

}

int main() {
	// 在主机内存中定义输入数据
	uint32_t a[SIZE];
	uint32_t b[SIZE];
	init<<<1,1>>>();
	
	for(size_t count=0;count<1;count++){
		srand(count);
		for(size_t j=0;j<SIZE;j++){
			a[j]=(rand()<<16)|(rand());
			b[j]=(rand()<<16)|(rand());
		}

		uint32_t c[SIZE] = { 
		0xf0000001, 0x43e1f593, 
		0x79b97091, 0x2833e848, 
		0x8181585d, 0xb85045b6, 
		0xe131a029, 0x30644e72};
		uint32_t r[SIZE];
		uint32_t r_[SIZE];
	
		mpz_t A, B, C, R;

		mpz_init(A); mpz_init(B); mpz_init(C); mpz_init(R);
		mpz_import(A, SIZE, -1, sizeof(uint32_t), 0, 0, a);
		mpz_import(B, SIZE, -1, sizeof(uint32_t), 0, 0, b);
		mpz_import(C, SIZE, -1, sizeof(uint32_t), 0, 0, c);
		for(int i=0;i<64;i++){
			mpz_mul(R, A, B);
			mpz_mod(R, R, C);
		}

		//gmp_printf("Result: %Zd\n", R);
		size_t mark;
		mpz_export(r, &mark, -1, sizeof(uint32_t), 0, 0, R);
		while (mark<SIZE)
			r[mark++] = 0;
		/*
		cout << "GMP:\n";
		//cout<<"MARK:"<<mark<<endl;
		for (int i = 0; i<SIZE; i++)
		{
			cout << hex << r[i] << endl;
		}*/
	
		to_mont(a,A,C);
		to_mont(b,B,C);
	
		uint32_t* d_a;
		uint32_t* d_b;
		uint32_t* d_c;
		cudaMalloc(&d_a, sizeof(uint32_t)*SIZE);
		cudaMalloc(&d_b, sizeof(uint32_t)*SIZE);
		cudaMalloc(&d_c, sizeof(uint32_t)*SIZE);
		cudaMemcpy(d_a, a, sizeof(uint32_t)*SIZE, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, sizeof(uint32_t)*SIZE, cudaMemcpyHostToDevice);

		mul_kernel<SIZE> <<<4096,512>>>(d_a, d_b, d_c);
		cudaDeviceSynchronize();
		float elapsedTime;
		cudaEvent_t e_start, e_stop;
		cudaEventCreate(&e_start);
		cudaEventCreate(&e_stop);
		cudaEventRecord(e_start, 0);
		mul_kernel<SIZE> <<<4096,512>>>(d_a, d_b, d_c);
		cudaDeviceSynchronize();
		cudaEventRecord(e_stop, 0);
		cudaEventSynchronize(e_stop);
		cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
		printf("mul time:%.4f ms\n", elapsedTime);
		elapsedTime=0;

		add_kernel<SIZE> <<<4096,512>>>(d_a, d_b, d_c);
		cudaDeviceSynchronize();
		cudaEventCreate(&e_start);
		cudaEventCreate(&e_stop);
		cudaEventRecord(e_start, 0);
		add_kernel<SIZE> <<<4096,512>>>(d_a, d_b, d_c);
		cudaDeviceSynchronize();
		cudaEventRecord(e_stop, 0);
		cudaEventSynchronize(e_stop);
		cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
		printf("add time:%.4f ms\n", elapsedTime);
		elapsedTime=0;

		sub_kernel<SIZE> <<<4096,512>>>(d_a, d_b, d_c);
		cudaDeviceSynchronize();
		cudaEventCreate(&e_start);
		cudaEventCreate(&e_stop);
		cudaEventRecord(e_start, 0);
		sub_kernel<SIZE> <<<4096,512>>>(d_a, d_b, d_c);
		cudaDeviceSynchronize();
		cudaEventRecord(e_stop, 0);
		cudaEventSynchronize(e_stop);
		cudaEventElapsedTime(&elapsedTime, e_start, e_stop);
		printf("sub time:%.4f ms\n", elapsedTime);

		//cudaMemcpy(r_, d_c, sizeof(uint32_t)*SIZE, cudaMemcpyDeviceToHost);

		//mpz_import(A, SIZE, -1, sizeof(uint32_t), 0, 0, r_);
		//mont_back(r_,A,C);
		/*
		cout << "test:\n";
		for (int i = 0; i<SIZE; i++)
		{
			cout << hex << r_[i] << endl;
		}*/
		/*
		for (int i = 0; i<SIZE; i++)
		{

			if(r_[i]!=r[i]){
				printf("error!\n");
				return -1;
			}
		}
		*/
	}
	
	printf("success!\n");
	return 0;
}