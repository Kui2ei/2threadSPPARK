// project begined at 20240411 20：05
// author:Bufanzhen
// https://github.com/Kui2ei
// project is on the basis SPPARK AND CGBN

// LIMITATION:only 2 thread, like thread1 and thread2 compute the 
// same modular multiplication, thread3 and thread4 comp......

// OUTPUT and INPUT:thread1 storage the first half of A,B,MOD,RESULT, thread2 storge
// the other half.
// RESULT is the OUTPUT ,also be seperated as 2 half.


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

//  First implement 2 thread
const int TPI = 2;
// const int tpi = 2;
// First implement 128bit
// const int bit = 128;
const int LIMBS = 2;


// 与原版不同，此函数只会左移1个单元
static inline void __device__ madc_n_rshift1(uint32_t* odd, const uint32_t *a, uint32_t bi,uint32_t odd_next)
{
        int n=LIMBS;
        for (size_t j = 0; j < n-2; j += 2)
            asm("madc.lo.cc.u32 %0, %2, %3, %1; madc.hi.cc.u32 %1, %2, %3, %4;"
                : "=r"(odd[j]), "+r"(odd[j+1])
                : "r"(a[j]), "r"(bi), "r"(odd[j+2]));
        asm("madc.lo.cc.u32 %0, %2, %3, %1; madc.hi.u32 %1, %2, %3, %4;"
            : "=r"(odd[n-2]), "+r"(odd[n-1])
            : "r"(a[n-2]), "r"(bi), "r"(odd_next));
}


__device__ __forceinline__ void  mont_mul(uint32_t r[LIMBS], const uint32_t a[LIMBS], 
    const uint32_t b[LIMBS], const uint32_t n[LIMBS], const uint32_t np0)
{
    uint32_t odd[LIMBS];
    uint32_t oddHighest = 0; //对应odd的最高位
    uint32_t oddHighestPlus1 = 0; //对应最高位的后一位

    uint32_t group_thread=threadIdx.x & TPI-1;
    uint32_t group_ID=threadIdx.x>>1;
    // uint32_t group_ID=threadIdx.x;
    // uint32_t group_thread=threadIdx.x & 1; //if 2 thread
    for(int i = 0;i <2;i++){
        for(int j = 0; j<LIMBS;j++){
            uint32_t bi = __shfl_sync(0xffffffff, b[j], (group_ID<<1)+i, TPI);
            if(i==0&&j==0){
                mul_n<LIMBS>(odd, a+1, bi);
                mul_n<LIMBS>(r  , a  , bi);
                
            }
            else{
                
                // 下面两步骤是将odd的最低位加到even的第二低位中，这样odd和even 都可以左移了
                // if(group_thread==0){
                // asm("addc.cc.u32 %0, %0, %1;" : "+r"(r[1]) : "r"(odd[0]));
                // }
                uint32_t temp_odd_0 = odd[0]; //把odd[0]存起来，然后在左移完之后加到r[0]中，完美
                uint32_t odd_next = 0;
                uint32_t even_next = 0;
                uint32_t odd_carry = 0;


                // 其实只需要thread0需要下面两步，thread1需要将这两个变量置为0
                // edi2，现在odd也不需要利用上一级的odd的最低为来帮助左移了，
                // 本身的odd[0]被存储好，左移变成最低位，加到r[0]
                // odd_next = __shfl_sync(0xffffffff, odd[0], (group_ID<<1)+1, TPI);
                even_next = __shfl_sync(0xffffffff, r[0], (group_ID<<1)+1, TPI);
                // if(i==0&&j==1){
                //     printf("THREAD:%d;i = %d;j = %d;odd_next = %x\n",threadIdx.x,i,j,odd_next);
                //     printf("THREAD:%d;i = %d;j = %d;even_next = %x\n",threadIdx.x,i,j,even_next);
                //     }
                if(group_thread==1){
                    // odd_next = 0;
                    even_next = 0;
                }
                if(i==0&&j==1){
                printf("THREAD:%d;i = %d;j = %d;odd_next = %x\n",threadIdx.x,i,j,odd_next);
                printf("THREAD:%d;i = %d;j = %d;even_next = %x\n",threadIdx.x,i,j,even_next);
                }
                madc_n_rshift1(r, a, bi, even_next);
                asm("addc.cc.u32 %0, %0, 0;" : "+r"(oddHighestPlus1));
                madc_n_rshift1(odd, a+1, bi, odd_next);
                // 这里在第二个线程的时候会出现一个进位。奇怪，明明第二个线程的时候肯定不会溢出
                asm("addc.cc.u32 %0, %0, 0;" : "+r"(odd_carry));
                // printf("\n");
                // if(i==1&&j==0){
                //     printf("THREAD:%d;i = %d;j = %d;odd[0] = %x\n",threadIdx.x,i,j,odd[0]);
                //     printf("THREAD:%d;i = %d;j = %d;odd[1] = %x\n",threadIdx.x,i,j,odd[1]);
                //     printf("THREAD:%d;i = %d;j = %d;r[0] = %x\n",threadIdx.x,i,j,r[0]);
                //     printf("THREAD:%d;i = %d;j = %d;r[1] = %x\n",threadIdx.x,i,j,r[1]);
                //     }

                asm("addc.cc.u32 %0, %0, %1;" : "+r"(r[LIMBS-1]) : "r"(oddHighest));
                asm("addc.cc.u32 %0, %0, %1;" : "+r"(odd[LIMBS-1]) : "r"(oddHighestPlus1));
                if(group_thread==0){
                    asm("addc.cc.u32 %0, %0, 0;" : "+r"(odd_carry) );
                }
                else{
                    odd_carry = 0;
                }
                odd_carry = __shfl_xor_sync(0xffffffff, odd_carry, (group_ID<<1)+1, TPI);
                asm("addc.cc.u32 %0, %0, %1;" : "+r"(odd[0]) : "r"(odd_carry));
                for(int k =1;k<LIMBS;k++){
                    asm("addc.cc.u32 %0, %0, 0;" : "+r"(odd[k]));
                }
                
                asm("addc.cc.u32 %0, %0, %1;" : "+r"(r[0]) : "r"(temp_odd_0));
                for(int k =1;k<LIMBS;k++){
                    asm("addc.cc.u32 %0, %0, 0;" : "+r"(r[k]));
                }
                printf("\n");
                if(i==0&&j==1){
                    printf("THREAD:%d;i = %d;j = %d;odd[0] = %x\n",threadIdx.x,i,j,odd[0]);
                    printf("THREAD:%d;i = %d;j = %d;odd[1] = %x\n",threadIdx.x,i,j,odd[1]);
                    printf("THREAD:%d;i = %d;j = %d;r[0] = %x\n",threadIdx.x,i,j,r[0]);
                    printf("THREAD:%d;i = %d;j = %d;r[1] = %x\n",threadIdx.x,i,j,r[1]);
                    }
                
                    oddHighest= 0;
                    oddHighestPlus1 = 0;
            }
        






            // 下面开始约简=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            uint32_t even0 = __shfl_sync(0xffffffff, r[0], (group_ID<<1), TPI);
            uint32_t mi = np0*even0;
            // printf("THREAD:%d;i = %d;j = %d;mi = %x\n",threadIdx.x,i,j,mi);
            // printf("\n");
            //     if(i==0&&j==1){
            //         printf("THREAD:%d;i = %d;j = %d;odd[0] = %x\n",threadIdx.x,i,j,odd[0]);
            //         printf("THREAD:%d;i = %d;j = %d;odd[1] = %x\n",threadIdx.x,i,j,odd[1]);
            //         printf("THREAD:%d;i = %d;j = %d;r[0] = %x\n",threadIdx.x,i,j,r[0]);
            //         printf("THREAD:%d;i = %d;j = %d;r[1] = %x\n",threadIdx.x,i,j,r[1]);
            //         }
            // printf("\n");
            //     if(i==0&&j==1){
            //         printf("THREAD:%d_________oddHighest = %x;oddHighestplus1 = %x\n",threadIdx.x,oddHighest,oddHighestPlus1);
            //         printf("THREAD:%d;i = %d;j = %d;odd[0] = %x\n",threadIdx.x,i,j,odd[0]);
            //         printf("THREAD:%d;i = %d;j = %d;odd[1] = %x\n",threadIdx.x,i,j,odd[1]);
            //         printf("THREAD:%d;i = %d;j = %d;r[0] = %x\n",threadIdx.x,i,j,r[0]);
            //         printf("THREAD:%d;i = %d;j = %d;r[1] = %x\n",threadIdx.x,i,j,r[1]);
            //         }
            cmad_n<LIMBS>(r, n,  mi);
            asm("addc.u32 %0, %0, 0;" : "+r"(oddHighest)); //将even的溢出存储在odd_1中，也就是对应着本单元内的最高位
            cmad_n<LIMBS>(odd, n+1, mi);
            asm("addc.u32 %0, %0, 0;" : "+r"(oddHighestPlus1));
            // 进位问题 待解决，在这里的每个线程都有可能发生溢出，
            // 具体来说ODD代表一个线程中所能表示的最高位，但是会发生溢出。
            // 在第一次实现中我们先假设并不是32的倍数的位数情况，也就是说
            // 最后一个线程（在这里也就是最后一个线程）一定不会发生溢出。
            // 问题简化成了，第一个线程这里产生进位，进位要进到第二个线程的
            // 第二低的位置，也就是ODD的最低位置，EVEN的第二低的位置
            
        }

    }

    // 最后还会剩下oddHighestPlus1两个变量没有处理（未完成）
    // merge
    uint32_t tmp = 0;
   
    // 这里也是只需要thread0需要，但是thread1对此无所谓，所以不需要进一步的设置
    tmp =  __shfl_xor_sync(0xffffffff, r[0], (group_ID<<1)+1, TPI);
    for(int i = 0; i<LIMBS-1;i++){
        asm("addc.cc.u32 %0, %1, %2;" : "+r"(r[i]) : "r"(r[i+1]), "r"(odd[i]));
    }
    
    if(group_thread==0){
        asm("addc.cc.u32 %0, %1, %2;" : "+r"(r[LIMBS-1]) : "r"(tmp), "r"(odd[LIMBS-1]));
        asm("addc.cc.u32 %0, %0, 0;" : "+r"(tmp) : "r"(tmp));
    }
    else{
        asm("addc.cc.u32 %0, %1, 0;" : "+r"(r[LIMBS-1]) : "r"(odd[LIMBS-1]));
        asm("addc.cc.u32 %0, %0, %1;" : "+r"(r[0]) : "r"(tmp));
        for(int i =1;i<LIMBS;i++){
            asm("addc.cc.u32 %0, %0, 0;" : "+r"(r[i]) : "r"(r[i]));
        }
    }

   
    
}

__global__ void runtest(uint32_t r[LIMBS*2]){
    const uint32_t a[LIMBS*2] = {0x11111111,0x22222222,0x33333333,0x44444444};
    const uint32_t b[LIMBS*2] = {0x55555555,0x66666666,0x77777777,0x88888888};
    
    const uint32_t MOD[LIMBS*2] = {0x99999999,0xaaaaaaaa,0xbbbbbbbb,0xcccccccc};
    const uint32_t np0 = 0x55555557;
    mont_mul(r+threadIdx.x*LIMBS,a+threadIdx.x*LIMBS,b+threadIdx.x*LIMBS,MOD+threadIdx.x*LIMBS,np0);
}

int main(){
    uint32_t r[LIMBS*2] = {0x0,0x0,0x0,0x0};
    uint32_t* d_r;
    cudaMalloc((void**)&d_r, sizeof(uint32_t)*4);
    cudaMemcpy(d_r, r, sizeof(uint32_t)*4, cudaMemcpyHostToDevice);
    runtest<<<1,2>>>(d_r);
    cudaDeviceSynchronize();
    cudaMemcpy(r, d_r, sizeof(uint32_t)*4, cudaMemcpyDeviceToHost);
    for(int i = 0;i<LIMBS*2;i++){ 
        printf("r[%d] : %x:",i,r[i]);
    }
    return 0 ;
}