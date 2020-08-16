namespace __polar_detail {
struct _gentype {
    _gentype();
    _gentype(int);
    _gentype(long);
    _gentype(unsigned);
    _gentype(unsigned long);
    _gentype(float);
    _gentype(double);
    _gentype& operator+(const _gentype&);
    _gentype& operator-(const _gentype&);
    _gentype& operator/(const _gentype&);
    _gentype& operator*(const _gentype&);
    _gentype& operator%(const _gentype&);
    _gentype& operator=(const _gentype&);
    _gentype& operator=(int);
    _gentype& operator=(long);
    _gentype& operator=(unsigned);
    _gentype& operator=(unsigned long);
    _gentype& operator=(double);
    _gentype& operator=(float);
    _gentype& operator+=(const _gentype&);
    _gentype& operator-=(const _gentype&);
    _gentype& operator/=(const _gentype&);
    _gentype& operator*=(const _gentype&);
    _gentype& operator<<(const _gentype&);
    _gentype& operator>>(const _gentype&);
};

struct __vec {
    __vec();
    __vec(int);
    __vec(long);
    __vec(unsigned);
    __vec(unsigned long);
    __vec(float);
    __vec(double);
    __vec& operator+(const __vec&);
    __vec& operator-(const __vec&);
    __vec& operator/(const __vec&);
    __vec& operator*(const __vec&);
    __vec& operator%(const __vec&);
    __vec& operator=(const __vec&);
    __vec& operator=(int);
    __vec& operator=(long);
    __vec& operator=(unsigned);
    __vec& operator=(unsigned long);
    __vec& operator=(double);
    __vec& operator=(float);
    __vec& operator+=(const __vec&);
    __vec& operator-=(const __vec&);
    __vec& operator/=(const __vec&);
    __vec& operator*=(const __vec&);
    __vec& operator<<(const __vec&);
    __vec& operator>>(const __vec&);
};

struct __matrix {};
}

using gentype = __polar_detail::_gentype;

template <int>
using vec = __polar_detail::__vec;

template <int, int>
using matrix = __polar_detail::__matrix;
