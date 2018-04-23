#ifndef RANDOM_H
#define RANDOM_H

extern void init_genrand(unsigned long s);
extern void init_by_array(unsigned long init_key[], int key_length);
extern unsigned long genrand_int32(void);
extern double genrand_real2(void);
extern double normal_cdf(double x);

#endif
