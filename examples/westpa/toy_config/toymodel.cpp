#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <random>


namespace py = pybind11;

static std::mt19937 rng;

static float sigma(float x) 
{
    return 0.5*tanh(4.0*(x - 0.5)) + 0.5;
}


static float asigma(float x) 
{
    return -0.5*tanh(4.0*(x - 0.5)) + 0.5;
}


/* model potential for double well, modeled with a spline
 * delta : free energy difference between wells
 * h : barrier height
 * */
static float spline(float x, float delta, float h) 
{
    if (std::isinf(delta))
        return 12*h*x*x;
    if (x<0.5) {
        if (x<0) {
            return -16*h*x*x*x + 12*h*x*x;
        } else {
            return -16*h*x*x*x + 12*h*x*x;
        }
    } else {
        /* TODO: check for negative delta; TODO: is there any better definition */
        /* TODO: yes let the barrier height be max(h, delta+h) */
        float y = 1.0 - x;
        if (x<1) {
            return -16*(h-delta)*y*y*y + 12*(h-delta)*y*y + delta;
        } else {
            return -16*(h+delta)*y*y*y + 12*fabs(h-delta)*y*y + delta;
        }
    }
    
}


float energy(size_t N, const float * __restrict__ x, 
              const float * const __restrict__ * __restrict__ delta,
              const float * const __restrict__ * __restrict__ h) 
{
    float energy = 0.0;
    for (size_t n=0u; n<N; n++) {
        float sum = 0.0;
        /* in summation over the multi index I, every index i is a binary digit of I */
        for(size_t I=0u; I < (1u<<n); I++) {
            float factor = exp(-spline(x[n], delta[n][I], h[n][I]));
            if (n>0) {
                for (size_t k=0u; k<n; k++) { /* move this loop up for optimization? */
                    size_t i = (I>>k) & 1u;
                    if (i==0) {
                        factor *= asigma(x[k]); /* ^(1-i) */ /* 1-sigma(x[k]) */
                    } else { 
                        factor *= sigma(x[k]);  /* ^i */
                    }
                }
            }
            sum += factor;
        }
        energy -= log(sum);
    }
    return energy;
}

static const float **unpack_params(py::list &p)
{
    auto M = static_cast<size_t>(p.size());
    const float ** p_line = new const float*[M]; // TODO: shared_pointer?
    size_t i = 0;
    for (auto p_line_obj : p) {
        if (!py::isinstance<py::array_t<float>>(p_line_obj))
            throw std::runtime_error("potential parameters has the wrong floating point type, must be np.float32");
        py::array_t<float> p_line_np = p_line_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast> >();
        if (p_line_np.ndim() != 1) {
            throw std::runtime_error("elements of parameter list must by one-dimensional");
        }
        auto n = static_cast<size_t>(p_line_np.shape(0));
        if (n!=(1u<<i)) throw std::runtime_error(std::string("parameters are not well shaped "));// + std::to_string(n) + " " + std::to_string(i) + " " + std::to_string(1<<i));
        auto p_line_raw = p_line_np.template unchecked<1>();
        // TODO: check that pointers don't alias each other
        p_line[i] = p_line_raw.data(0);
        i++;
    }

    return p_line;
}

float numpy_energy(py::array_t<float, py::array::c_style | py::array::forcecast> &np_x, 
                    py::list &delta, py::list &h)
{
    auto M = static_cast<size_t>(delta.size());
    auto K = static_cast<size_t>(h.size());
    if (M!=K) throw std::runtime_error("dimensions of delta and h must match");
    const float ** delta_ptr = unpack_params(delta); /* todo smart ptr and move */
    const float ** h_ptr = unpack_params(h);

    if (np_x.ndim() != 1) {
        throw std::runtime_error("x must by one-dimensional");
    }
    auto N = static_cast<size_t>(np_x.shape(0));
    if (N!=M) throw std::runtime_error("dimensions of potential (delta, h) and x must match");
    auto x = np_x.template mutable_unchecked<1>();

    float ener = energy(N, x.mutable_data(0), delta_ptr, h_ptr);
    delete[] delta_ptr;
    delete[] h_ptr;
    return ener;
}

void propagate_mcmc(unsigned int n_steps,
                    unsigned int N, float * __restrict__ x,
                    float * __restrict__ buffer, 
                    const float * const __restrict__ * __restrict__ delta,
                    const float * const __restrict__ * __restrict__ h,
                    float step_size,
                    int kinetic
                    /*StringWeb* meta*/)
{
    float *y = buffer;
    float e, e_prime;
    std::uniform_real_distribution<float> distr_u(0, 1);
    std::uniform_int_distribution<unsigned int> distr_01(0, 1);

    for(size_t t=0; t<n_steps; ++t) {
        e = energy(N, x, delta, h);
        /*if (meta) e += meta->energy(x);*/
        if (!kinetic && random() > 0.5) {
            /* attempt a large step */
            for(size_t i=0; i<N; ++i) { 
                y[i] = fmod(x[i] + distr_01(rng), 2.0); /* pick lower dimensions more frequently? */
            }
        } else {
            /* attempt a local step */
            for(size_t i=0; i<N; ++i) {
                y[i] = x[i] + step_size*(2*distr_u(rng) - 1.0);
            }
        }
        e_prime = energy(N, y, delta, h);
        /*if (meta) e_prime += meta->energy(y);*/

        if (exp(e-e_prime) > distr_u(rng)) { /* accepted */
            float *tmp = y;
            y = x;
            x = tmp;
        }
    }
    /* current configuration is already in output buffer? */
    if (x!=buffer) memcpy(buffer, x, N*sizeof(float));
}

py::array_t<float> numpy_propagate(unsigned int n_steps,
                       py::array_t<float, py::array::c_style | py::array::forcecast> &np_x,
                       py::list &delta, py::list &h,
                       py::array_t<float, py::array::c_style | py::array::forcecast> &np_out,
                       unsigned int stride,
                       bool kinetic,
                       float step_size,
                       unsigned int rng_seed)
                       /*std::string metadynamics*/
{
    auto M = static_cast<size_t>(delta.size());
    auto K = static_cast<size_t>(h.size());
    if (M!=K) throw std::runtime_error("dimensions of delta and h must match");
    const float ** delta_ptr = unpack_params(delta); /* todo smart ptr and move */
    const float ** h_ptr = unpack_params(h);

    if (!py::isinstance<py::array_t<float>>(np_x))
        throw std::runtime_error("initial position vector has the wrong floating point type, must be np.float32");
    if (!py::isinstance<py::array_t<float>>(np_out))
        throw std::runtime_error("output trajectory buffer has the wrong floating point type, must be np.float32");
    if (np_x.ndim() != 1) {
        throw std::runtime_error("x must by one-dimensional");
    }
    auto N = static_cast<size_t>(np_x.shape(0));
    if (N!=M) throw std::runtime_error("dimensions of potential (delta, h) and x must match");
    auto pos = np_x.template mutable_unchecked<1>();

    if (np_out.ndim() != 2) {
        throw std::runtime_error("out must by two-dimensional (matrix)");
    }
    auto O = static_cast<size_t>(np_out.shape(1));
    if (np_out.shape(0) < 2) {
        throw std::runtime_error("shape[0] of out must be at least 2");
    }
    if (n_steps >= stride * static_cast<size_t>(np_out.shape(0))) 
        throw std::runtime_error("number of integration steps is larger than the size of out");
    if (O!=M) throw std::runtime_error("dimensions of out and x must match");
    auto out = np_out.template mutable_unchecked<2>();

    if  (0!=rng_seed)
        rng.seed(rng_seed);

    memcpy(out.mutable_data(0, 0), pos.mutable_data(0), N*sizeof(float));
    float *scratch = new float[N];
    for(size_t t=0; t<n_steps; t++) {
        memcpy(scratch, out.mutable_data(t, 0), N*sizeof(float));
        propagate_mcmc(stride, N, scratch, out.mutable_data(t+1, 0), delta_ptr, h_ptr, step_size, kinetic/*, NULL*/);
    }

    /* TODO: create out buffer if not given */
    /*  auto result = py::array_t<double>(buf1.size); */

    delete[] scratch;
    delete[] delta_ptr;
    delete[] h_ptr;

    return np_out;
}

PYBIND11_MODULE(toymodel, m) {
    m.doc() = "monsterwell model";
    m.def("energy", &numpy_energy, "get model energy", py::arg("x"), py::arg("delta"), py::arg("h"));
    m.def("propagate", &numpy_propagate, "propagate with MCMC", py::arg("n_steps"), py::arg("x"), py::arg("delta"),
          py::arg("h"), py::arg("out").none(false), py::arg("stride")=1, py::arg("kinetic")=true, py::arg("step_size")=0.05, py::arg("rng_seed")=0);
    rng.seed(std::random_device()());
}
