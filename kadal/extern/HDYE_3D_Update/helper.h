#ifndef EHVI_HELPER_H
#define EHVI_HELPER_H
#include "ehvi_consts.h"
#include <math.h>
#include <memory>

//Header of the helper functions used for expected hypervolume calculations.
//Also contains the individual struct.

static inline double erffast(double x)
{
    // constants
    double a1 =  0.254829592;
    double a2 = -0.284496736;
    double a3 =  1.421413741;
    double a4 = -1.453152027;
    double a5 =  1.061405429;
    double p  =  0.3275911;

    // Save the sign of x
    int sign = 1;
    if (x < 0)
        sign = -1;
    x = fabs(x);

    // A&S formula 7.1.26
    double t = 1.0/(1.0 + p*x);
    double y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*exp(-x*x);

    return sign*y;
}


//Individual struct. Holds one individual.
struct individual{
  double f[DIMENSIONS];
};

//This struct holds the heightmaps and an array of dominated hypervolumes
//used in the cell calculations for the current z value.
//To update chunk, all values of slice are multiplied by height
//and added to it. Updating slice uses some geometry, see paper.
struct thingy{
  double slice; //area covered by z-slice
  double chunk; //S^-
  int highestdominator; //highest z coordinate of point dominating the point.
  double xlim,ylim; //1-dimensional limits (zlim is calculated from highestdominator)  
};

//for convenient calculation of height maps.
struct specialind{
  std::shared_ptr<individual> point; //the individual's basic stats.
  /* individual *point; //the individual's basic stats. */
  int xorder, yorder, zorder; //Position of the individual in its respective sorting lists
};


//Comparison functions for sorting the specialind struct in y and z dimensions.
static inline bool specialycomparator(std::shared_ptr<specialind> A, std::shared_ptr<specialind> B){
/* static inline bool specialycomparator(specialind * A, specialind * B){ */
 return A->point->f[1] < B->point->f[1];
}

static inline bool specialzcomparator(std::shared_ptr<specialind> A, std::shared_ptr<specialind> B){
/* static inline bool specialzcomparator(specialind * A, specialind * B){ */
  return A->point->f[2] < B->point->f[2];
}

//Probability density function for the normal distribution.
static inline double gausspdf(double x){
  return SQRT_TWOPI_NEG * exp(-(x*x)/2);
}

//Cumulative distribution function for the normal distribution
static inline double gausscdf(double x){
  return 0.5*(1+erf(x/SQRT_TWO));
}

//Partial expected improvement function 'psi'.
static inline double exipsi(double fmax, double cellcorner, double mu, double s){
  return (s * gausspdf((cellcorner-mu)/s)) + ((fmax-mu) * gausscdf((cellcorner-mu)/s));
}

//Comparator function for sorting the inviduals in ascending order of x coordinate.
static inline bool xcomparator(std::shared_ptr<individual> A, std::shared_ptr<individual> B){
  return A->f[0] < B->f[0];
}

//Comparator function for sorting the inviduals in ascending order of y coordinate.
static inline bool ycomparator(std::shared_ptr<individual> A, std::shared_ptr<individual> B){
  return A->f[1] < B->f[1];
}

//Comparator function for sorting the inviduals in ascending order of z coordinate.
static inline bool zcomparator(std::shared_ptr<individual> A, std::shared_ptr<individual> B){
  return A->f[2] < B->f[2];
}/* 
//compares specialind in sort function.
bool specialycomparator(specialind * A, specialind * B);
bool specialzcomparator(specialind * A, specialind * B);

//Probability density function for the normal distribution.
double gausspdf(double x);

//Cumulative distribution function for the normal distribution
double gausscdf(double x);

//Partial expected improvement function 'psi'.
double exipsi(double fmax, double cellcorner, double mu, double s);

//Comparator function for sorting the inviduals in ascending order of x coordinate.
bool xcomparator(individual * A, individual * B);

//Comparator function for sorting the inviduals in ascending order of y coordinate.
bool ycomparator(individual * A, individual * B);

//Comparator function for sorting the inviduals in ascending order of z coordinate.
bool zcomparator(individual * A, individual * B); */
#endif
