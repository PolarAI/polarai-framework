namespace __polar_detail {
struct __blockIdx {
  unsigned x;
  unsigned y;
  unsigned z;
};
struct __blockDim {
  unsigned x;
  unsigned y;
  unsigned z;
};
struct __threadIdx {
  unsigned x;
  unsigned y;
  unsigned z;
};
struct __gridDim {
  unsigned x;
  unsigned y;
  unsigned z;
};
}

__polar_detail::__blockIdx blockIdx;
__polar_detail::__blockIdx blockDim;
__polar_detail::__blockIdx threadIdx;
__polar_detail::__blockIdx gridDim;
