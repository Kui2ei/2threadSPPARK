#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"




__device__ uint32_t M0=0;

__device__ __forceinline__ void copyArrayValues(uint32_t* dest, uint32_t* src) {
    #pragma unroll
    for (int i = 0; i < 8; i+=4) {
        *(__int128*)&dest[i] = *(__int128*)&src[i];
    }
    /*
    for (int i = 0; i < NUM; i++) {
        dest[i] = src[i];
    }*/
}

__device__ __forceinline__ uint32_t computeM0(uint32_t x) {
    uint32_t inv=x;
  
    inv=inv*(inv*x+14);
    inv=inv*(inv*x+2);
    inv=inv*(inv*x+2);
    inv=inv*(inv*x+2);
    return inv;
  }

template<uint32_t n>
static __device__ __forceinline__ void cadd_n(uint32_t* acc, uint32_t* a)
{
    asm("add.cc.u32 %0, %0, %1;" : "+r"(acc[0]) : "r"(a[0]));
    for (size_t i = 1; i < n; i++)
        asm("addc.cc.u32 %0, %0, %1;" : "+r"(acc[i]) : "r"(a[i]));
    // return carry flag
}

template<uint32_t n>
static __device__ __forceinline__ void final_subc(uint32_t* acc, uint32_t* MOD)
{
    uint32_t carry, tmp[n];

    asm("addc.u32 %0, 0, 0;" : "=r"(carry));

    asm("sub.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(acc[0]), "r"(MOD[0]));
    for (size_t i = 1; i < n; i++)
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(acc[i]), "r"(MOD[i]));
    asm("subc.u32 %0, %0, 0;" : "+r"(carry));

    asm("{ .reg.pred %top;");
    asm("setp.eq.u32 %top, %0, 0;" :: "r"(carry));
    for (size_t i = 0; i < n; i++)
        asm("@%top mov.b32 %0, %1;" : "+r"(acc[i]) : "r"(tmp[i]));
    asm("}");
}

template<uint32_t n>
static __device__ __forceinline__ void mul_n(uint32_t* acc, const uint32_t* a, uint32_t bi)
{
    for (size_t j = 0; j < n; j += 2)
        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
        : "=r"(acc[j]), "=r"(acc[j+1])
        : "r"(a[j]), "r"(bi));
}

template<uint32_t n>
static __device__ __forceinline__ void cmad_n(uint32_t* acc, const uint32_t* a, uint32_t bi)
{
    asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
    : "+r"(acc[0]), "+r"(acc[1])
    : "r"(a[0]), "r"(bi));
    for (size_t j = 2; j < n; j += 2)
        asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
        : "+r"(acc[j]), "+r"(acc[j+1])
        : "r"(a[j]), "r"(bi));
}

template<uint32_t n>
static __device__ __forceinline__ void madc_n_rshift(uint32_t* odd, const uint32_t *a, uint32_t bi)
{
    for (size_t j = 0; j < n-2; j += 2)
        asm("madc.lo.cc.u32 %0, %2, %3, %4; madc.hi.cc.u32 %1, %2, %3, %5;"
            : "=r"(odd[j]), "=r"(odd[j+1])
            : "r"(a[j]), "r"(bi), "r"(odd[j+2]), "r"(odd[j+3]));
    asm("madc.lo.cc.u32 %0, %2, %3, 0; madc.hi.u32 %1, %2, %3, 0;"
        : "=r"(odd[n-2]), "=r"(odd[n-1])
        : "r"(a[n-2]), "r"(bi));
}

template<uint32_t n>
static __device__ __forceinline__ void mad_n_redc(uint32_t *even, uint32_t* odd,
    const uint32_t *a, uint32_t bi, uint32_t* MOD, bool first=false)
{
    if (first) {
        mul_n<n>(odd, a+1, bi);
        mul_n<n>(even, a,  bi);
    } else {
        asm("add.cc.u32 %0, %0, %1;" : "+r"(even[0]) : "r"(odd[1]));
        madc_n_rshift<n>(odd, a+1, bi);
        cmad_n<n>(even, a, bi);
        asm("addc.u32 %0, %0, 0;" : "+r"(odd[n-1]));
    }

    uint32_t mi = even[0] * M0;

    cmad_n<n>(odd, MOD+1, mi);
    cmad_n<n>(even, MOD,  mi);
    asm("addc.u32 %0, %0, 0;" : "+r"(odd[n-1]));
}

template<uint32_t n>
__device__ __forceinline__ void final_sub(uint32_t carry, uint32_t* tmp, uint32_t* even, uint32_t* MOD, uint32_t N)
{
    size_t i;
    asm("{ .reg.pred %top;");

    asm("sub.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(even[0]), "r"(MOD[0]));
    for (i = 1; i < n; i++)
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(even[i]), "r"(MOD[i]));
    if (N%32 == 0)
        asm("subc.u32 %0, %0, 0; setp.eq.u32 %top, %0, 0;" : "+r"(carry));
    else
        asm("subc.u32 %0, 0, 0; setp.eq.u32 %top, %0, 0;" : "=r"(carry));

    for (i = 0; i < n; i++)
        asm("@%top mov.b32 %0, %1;" : "+r"(even[i]) : "r"(tmp[i]));

    asm("}");
}

template<uint32_t n>
__device__ __forceinline__ void operator_add(uint32_t* a, uint32_t* b, uint32_t* MOD, uint32_t* res)
{
    copyArrayValues(res,a);
    cadd_n<n>(&res[0], &b[0]);
    final_subc<n>(res, MOD);
}

template<uint32_t n>
__device__ __forceinline__ void operator_sub(uint32_t* a, uint32_t* b, uint32_t* MOD, uint32_t* res)
{
    size_t i;
    uint32_t tmp[n], borrow;
    copyArrayValues(res,a);

    asm("sub.cc.u32 %0, %0, %1;" : "+r"(res[0]) : "r"(b[0]));
    for (i = 1; i < n; i++)
        asm("subc.cc.u32 %0, %0, %1;" : "+r"(res[i]) : "r"(b[i]));
    asm("subc.u32 %0, 0, 0;" : "=r"(borrow));

    asm("add.cc.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(res[0]), "r"(MOD[0]));
    for (i = 1; i < n-1; i++)
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(res[i]), "r"(MOD[i]));
    asm("addc.u32 %0, %1, %2;" : "=r"(tmp[i]) : "r"(res[i]), "r"(MOD[i]));

    asm("{ .reg.pred %top; setp.ne.u32 %top, %0, 0;" :: "r"(borrow));
    for (i = 0; i < n; i++)
        asm("@%top mov.b32 %0, %1;" : "+r"(res[i]) : "r"(tmp[i]));
    asm("}");
}

template<uint32_t n>
__device__ __forceinline__ void operator_mul(uint32_t* a, uint32_t* b, uint32_t* MOD, uint32_t* res, uint32_t N)
{ 
    uint32_t even[n];
    uint32_t odd[n];
    #pragma unroll
    for (size_t i = 0; i < n; i += 2) {
        mad_n_redc<n>(&even[0], &odd[0], &a[0], b[i], &MOD[0], i==0);
        mad_n_redc<n>(&odd[0], &even[0], &a[0], b[i+1], &MOD[0]);
    }

    // merge |even| and |odd|
    cadd_n<n-1>(&even[0], &odd[1]);
    asm("addc.u32 %0, %0, 0;" : "+r"(even[n-1]));

    final_sub<n>(0, &odd[0], &even[0], &MOD[0], N);
    copyArrayValues(res,even);
}