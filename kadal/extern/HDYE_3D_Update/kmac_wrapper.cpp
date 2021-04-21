/**
 * @file kmac_wrapper.cpp
 *
 * @brief A wrapper for the KMAC 3D EHVI sliceupdate methods.
 *
 * @author Tim Jim
 * Contact: github.com/timjim333
 *
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include "helper.h"
#include "ehvi_calculations.h"
#include "ehvi_multi.h"
#include "ehvi_sliceupdate.h"

namespace py = pybind11;

// Copied from main.cc
//Checks if p dominates P. Removes points dominated by p from P and return the number of points removed.
int checkdominance(deque<shared_ptr<individual>> & P, shared_ptr<individual> p){
  int nr = 0;
  for (int i=P.size()-1;i>=0;i--){
    if (p->f[0] >= P[i]->f[0] && p->f[1] >= P[i]->f[1] && p->f[2] >= P[i]->f[2]){
      cerr << "Individual " << (i+1) << " is dominated or the same as another point; removing." << endl;
      P.erase(P.begin()+i);
      nr++;
    }
  }
  return nr;
}


// Wrapper to the single mean vector input ehvi3d_sliceupdate function
double wrap_ehvi3d_sliceupdate(py::array_t<double> y_par, py::array_t<double> ref_point, py::array_t<double> mean_vector, py::array_t<double> std_dev) {
  deque<shared_ptr<individual>> nd_samples;
  
  // Get y_par and feed by individual via numpy direct access
  // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
  auto yp = y_par.unchecked<2>(); // y_par must have ndim = 2
  
  for (py::ssize_t i = 0; i < yp.shape(0); i++) {
    auto tempvidual = make_shared<individual>();
    tempvidual->f[0] = yp(i, 0);
    tempvidual->f[1] = yp(i, 1);
    tempvidual->f[2] = yp(i, 2);
    // cerr << i << ": " << yp(i, 0) << " " << yp(i, 1) << " " << yp(i, 2) << endl;
    // cerr << i << ": " << tempvidual->f[0] << " " << tempvidual->f[1] << " " << tempvidual->f[2] << endl;
    checkdominance(nd_samples, tempvidual);
    nd_samples.push_back(tempvidual);
  }

  // Marshall ref_point, mean_vector, and std_dev into an array
  // (might be better ways to do this..)
  auto rp = ref_point.unchecked<1>(); // ref_point must have ndim = 1, len 3
  double r [] = {rp(0), rp(1), rp(2)};
 
  // If only 1 mean vector
  auto mv = mean_vector.unchecked<1>(); // mean_vector must have ndim = 1, len 3
  double mu [] = {mv(0), mv(1), mv(2)};
  auto sd = std_dev.unchecked<1>(); // std_dev must have ndim = 1, len 3
  double s [] = {sd(0), sd(1), sd(2)};
  double ehvi = ehvi3d_sliceupdate(nd_samples, r, mu, s);
  return ehvi;
}


// Wrapper to the multi mean vector input ehvi3d_sliceupdate function
py::array_t<double> wrap_ehvi3d_sliceupdate_multi(py::array_t<double> y_par, py::array_t<double> ref_point, py::array_t<double> mean_vector, py::array_t<double> std_dev) {
  deque<shared_ptr<individual>> nd_samples;
  
  // Get y_par and feed by individual via numpy direct access
  // https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
  auto yp = y_par.unchecked<2>(); // y_par must have ndim = 2
  
  for (py::ssize_t i = 0; i < yp.shape(0); i++) {
    auto tempvidual = make_shared<individual>();
    tempvidual->f[0] = yp(i, 0);
    tempvidual->f[1] = yp(i, 1);
    tempvidual->f[2] = yp(i, 2);
    // cerr << i << ": " << yp(i, 0) << " " << yp(i, 1) << " " << yp(i, 2) << endl;
    // cerr << i << ": " << tempvidual->f[0] << " " << tempvidual->f[1] << " " << tempvidual->f[2] << endl;
    checkdominance(nd_samples, tempvidual);
    nd_samples.push_back(tempvidual);
  }

  // Marshall ref_point, mean_vector, and std_dev into an array
  // (might be better ways to do this..)
  auto rp = ref_point.unchecked<1>(); // ref_point must have ndim = 1, len 3
  double r [] = {rp(0), rp(1), rp(2)};

  // Get dimensions of mean vector (assume std dev has same dims)
  py::buffer_info buf = mean_vector.request();
  vector<shared_ptr<mus>> pdf(buf.shape[0], make_shared<mus>());
  // Else if several mean vectors to test
  auto mv = mean_vector.unchecked<2>(); // mean_vector must have ndim = 2, len 3
  auto sd = std_dev.unchecked<2>(); // std_dev must have ndim = 2, len 3
  for (int i = 0; i < mv.shape(0); i++) {
    // pdf.push_back(make_shared<mus>());
    pdf[i] = make_shared<mus>();
    pdf[i]->mu[0] = mv(i, 0);
    pdf[i]->mu[1] = mv(i, 1);
    pdf[i]->mu[2] = mv(i, 2);
    pdf[i]->s[0] = sd(i, 0);
    pdf[i]->s[1] = sd(i, 1);
    pdf[i]->s[2] = sd(i, 2);
    // cerr << pdf[i]->mu[0] << pdf[i]->mu[1] << pdf[i]->mu[2] <<endl;
    // cerr << pdf[i]->s[0] << pdf[i]->s[1] << pdf[i]->s[2] << endl;
  }

  vector<double> ehvi = ehvi3d_sliceupdate(nd_samples, r, pdf);
  // for (double ehvi_i: ehvi)
  //   cerr << ehvi_i << endl;
  
  py::array ehvi_np = py::cast(ehvi);
  return ehvi_np;

  // Seems like memory use is stable without these now - above cast is faster atm
  // https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11/44682603#44682603
  // https://stackoverflow.com/questions/54876346/pybind11-and-stdvector-how-to-free-data-using-capsules
  // std::unique_ptr<std::vector<double>> result =
  //   std::make_unique<std::vector<double>>(ehvi3d_sliceupdate(nd_samples, r, pdf));
  // return py::array_t<double>({result->size()}, // shape
  // 			     {sizeof(double)}, // stride
  // 			     result->data());   // data pointer

  // vector<double> *result = new vector<double>(ehvi3d_sliceupdate(nd_samples, r, pdf));
  // py::capsule free_when_done(result, [](void *f) {
  // 				       auto foo = reinterpret_cast<vector<double> *>(f);
  // 				       delete foo;
  // 				     });
  // return py::array_t<double>({result->size()}, // shape
  // 			     {sizeof(double)}, // stride
  // 			     result->data(),   // data pointer
  // 			     free_when_done);
  
}


PYBIND11_MODULE(kmac, m) {
    // module docstring
    m.doc() = "EHVI using KMAC";

    // Write custom signatures
    py::options options;
    options.disable_function_signatures();

    // definie EHVI slice update function
    m.def("ehvi3d_sliceupdate", &wrap_ehvi3d_sliceupdate, R"(
    ehvi3d_sliceupdate(y_par, ref_point, mean_vector, std_dev)

    O(n^3) slice-update scheme for calculating the EHVI.

    n_obj must == 3. y_par array must be 2D.
    (If a n_pop == 1, ensure 2D using 'np.ndarray.reshape(1, -1)', 
    if necessary.)

    Args:
        y_par (np.ndarray): [n_pop, n_obj] 2D array of the current Pareto
            front.
        ref_point (np.ndarray): n_obj-len 1D array of the refrerence
            point.
        mean_vector (np.ndarray): n_obj-len 1D array of the sample
            mean vector.
        std_dev (np.ndarray): n_obj-len 1D array of the standard 
            deviation for the given mean vector.

    Returns:
        ehvi (float): The expected hypervolume improvement.
    
)");
    
    // definie EHVI slice multi update function
    m.def("ehvi3d_sliceupdate_multi", &wrap_ehvi3d_sliceupdate_multi, R"(
    ehvi3d_sliceupdate_multi(y_par, ref_point, mean_vector, std_dev)

    O(n^3) multi slice-update scheme for calculating the EHVI.

    n_obj must == 3. y_par, mean_vector and std_dev arrays must be 2D.
    (If a n_pop == 1, ensure 2D using 'np.ndarray.reshape(1, -1)', 
    if necessary.)

    Args:
        y_par (np.ndarray): [n_pop, n_obj] 2D array of the current Pareto
            front.
        ref_point (np.ndarray): n_obj-len 1D array of the refrerence
            point.
        mean_vector (np.ndarray): [n_pop, n_obj] 2D array of the sample
            mean vectors.
        std_dev (np.ndarray): [n_pop, n_obj] 2D array of the standard 
            deviation for the given mean vectors.

    Returns:
        ehvi (np.ndarray): Array of expected hypervolume improvements.
    
)");
    return;
}
