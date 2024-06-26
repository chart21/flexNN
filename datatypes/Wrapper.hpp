#include "../include/pch.h"
#define FRACTIONAL_VALUE 5



/* template <typename Datatype, typename float_type, typename INT_TYPE, typename UINT_TYPE, int fractional> */
/* void store_convert_vectorize(float val) */
/* { */
/*     UINT_TYPE fixVal = FloatFixedConverter<float_type, INT_TYPE, UINT_TYPE, fractional>::floatToFixed(val); */
/*     Datatype vecVal = PROMOTE(fixVal); */
/*     for(int i = 0; i < BITLENGTH; i++) */
/*     { */
/*         player_input[counter] = vecVal; */
/*         counter++; */
/*     } */
/* } */
/* template <typename Datatype, typename float_type, typename INT_TYPE, typename UINT_TYPE, int fractional> */
/* void store_convert_ortho(float val[BITLENGTH][DATTYPE/BITLENGTH]) */
/* { */
/*     alignas(sizeof(DATATYPE)) UINT_TYPE fixVals[BITLENGTH][DATTYPE/BITLENGTH]; */
/*     for(int i = 0; i < BITLENGTH; i++) */
/*     { */
/*         for(int j = 0; j < DATTYPE/BITLENGTH; j++) */
/*         { */
/*             fixVals[i][j] = FloatFixedConverter<float_type, INT_TYPE, UINT_TYPE, fractional>::floatToFixed(val[i][j]); */
/*         } */
/*     } */
/*     orthogonalize_arithmetic((UINT_TYPE*) fixVals, (Datatype*) fixVals); */
    
/*     for(int i = 0; i < BITLENGTH; i++) */
/*     { */
/*         player_input[counter] = ((Datatype*) fixVals)[i]; */
/*         counter++; */
/*     } */
/* } */

template <typename T>
T truncate(const T& val)
{
    return val;
}

template <>
float truncate(const float& val) {
    return val; // No truncation needed for float
}

template <>
uint8_t truncate(const uint8_t& val) {
    int8_t temp = static_cast<int8_t>(val);
    temp >>= FRACTIONAL_VALUE;
    return static_cast<uint8_t>(temp);
}

template <>
uint16_t truncate(const uint16_t& val) {
    int16_t temp = static_cast<int16_t>(val);
    temp >>= FRACTIONAL_VALUE;
    return static_cast<uint16_t>(temp);
}

template <>
uint32_t truncate(const uint32_t& val) {
    int32_t temp = static_cast<int32_t>(val);
    temp >>= FRACTIONAL_VALUE;
    return static_cast<uint32_t>(temp);
}

template <>
uint64_t truncate(const uint64_t& val) {
    int64_t temp = static_cast<int64_t>(val);
    temp >>= FRACTIONAL_VALUE;
    return static_cast<uint64_t>(temp);
}



template<typename float_type, typename INT_TYPE, typename UINT_TYPE, int fractional, typename T>
class Wrapper{
using W = Wrapper<float_type, INT_TYPE, UINT_TYPE, fractional, T>;
T s1;
    public:

Wrapper(float s)
{
    this->s1 = FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL_VALUE>::float_to_ufixed(s);
}

Wrapper(T s, int dummy)
{
    this->s1 = s;
}

Wrapper(){
this->s1 = 0;
}

T get_s1(){
    return this->s1;
}



Wrapper operator+(const Wrapper s) const{
    return Wrapper(this->s1 + s.s1, 0);
}

Wrapper operator-(const Wrapper s) const{
    return Wrapper(this->s1 - s.s1, 0);
}

Wrapper operator*(const Wrapper s) const{
    /* if(Wrapper(this->s1 * s.s1,0).reveal() > 1000000){ */
    /*     std::cout << "overflow" <<std::endl; */

    /* } */
    return Wrapper(this->s1 * s.s1, 0);
}

Wrapper operator*(const int s) const{
    return Wrapper(this->s1 * s, 0);
}

Wrapper operator/(const int s) const{
    return Wrapper(this->s1 / s, 0);
}

void mask_and_send_dot() 
{
    /* this->s1 = truncate(this->s1); */
    /* if(std::abs(this->reveal()) > 50){ */
    /*     std::cout << "s1 " << this->reveal()  << std::endl; */
    /* } */
}

void complete_mult()
{
    this->s1 = truncate(this->s1);

}

void operator/=(const int s){
    this->s1 /= s;
}

void operator+=(const Wrapper s){
    this->s1 += s.s1;
}

void operator-=(const Wrapper s){
    this->s1 -= s.s1;
}

void operator*= (const Wrapper s){
*this = *this * s;
}

bool operator==(const W& other) const {
    return false; 
}

//temporary solution for max pool
bool operator > (const W& other) const {
    if constexpr (std::is_same_v<T, float>) {
        return this->s1 > other.s1;
    }
    else
    {
    bool isNegative = (this->s1 & (T(1) << (sizeof(T)*8 - 1))) != 0;
    bool otherIsNegative = (other.s1 & (T(1) << (sizeof(T)*8 - 1))) != 0;
    if (isNegative && !otherIsNegative)
        return false;
    else if (!isNegative && otherIsNegative)
        return true;
    else if (isNegative && otherIsNegative)
        return this->s1 < other.s1;
    else
        return this->s1 > other.s1;
    }
    /* return this->s1 > other.s1; */
}

Wrapper relu() const{
    if constexpr (std::is_same_v<T, float>) {
        return Wrapper(this->s1 > 0 ? this->s1 : 0);
    }
    else
    {
    bool isNegative = (this->s1 & (T(1) << (sizeof(T)*8 - 1))) != 0;
    if (!isNegative)
        return Wrapper(this->s1,0);
    else
        return Wrapper(0,0);
    }
    /* return W(isNegative ? T(0) : this->s1); */    
    /* if (this->s1 > 0) */
    /*     return Wrapper(this->s1,0); */
    /* else */
    /*     return Wrapper(0,0); */
}

/* Wrapper<T> drelu() const{ */
/*     return Wrapper<T>(this->s1 > 0 ? Wrapper(1) : Wrapper(0)); */
/* } */

Wrapper drelu(const Wrapper& other) const{
    if (this->s1 > 0)
        return Wrapper(other.s1,0);
    else
        return Wrapper(0,0);
    /* return Wrapper(this->s1 > 0 ? other : 0,0); */
}

//temporary solutions for softmax

W operator/(const W& other) const{
    return W(this->s1 / other.s1, 0);
}

W exp() const{
    return W(std::exp(this->s1), 0);
}

W log() const{
    return W(std::log(this->s1),0);
}

// Recursive tree-based max computation
static W treeFindMax(const W* begin, const W* end) {
    // Base case: If there's only one element, return it
    if (end - begin == 1) {
        return *begin;
    }

    // Split data into two halves
    const W* mid = begin + (end - begin) / 2;

    // Recursively find max in each half
    W leftMax = treeFindMax(begin, mid);
    W rightMax = treeFindMax(mid, end);

    // Return the larger of the two max values
    return (leftMax > rightMax) ? leftMax : rightMax;
}

static void RELU(const W* begin, const W*end, W* output){
    int i = 0;
    for (const W* iter = begin; iter != end; ++iter) {
            output[i++] = iter->relu();
    }
}

static void communicate()
{
    /* std::cout << "communicate" << std::endl; */
}

static W findMax(const W* begin, const W* end) {
        W max_val = *begin;
        for (const W* iter = begin; iter != end; ++iter) {
            if (*iter > max_val) {
                max_val = *iter;
            }
        }
        return max_val;
    }

/* static std::ptrdiff_t argMax(const Wrapper<T>* begin, const Wrapper<T>* end) { */
/*     const Wrapper<T>* max_iter = begin; */
/*     for (const Wrapper<T>* iter = begin; iter != end; ++iter) { */
/*         if (*iter > *max_iter) { */
/*             max_iter = iter; */
/*         } */
/*     } */
/*     return std::distance(begin, max_iter); */
/* } */

static void argMax(const W* begin, const W* end, W* output) {
    const W* max_iter = begin;
    for (const W* iter = begin; iter != end; ++iter) {
        if (*iter > *max_iter) {
            max_iter = iter;
        }
    }
    std::ptrdiff_t max_idx = std::distance(begin, max_iter);
    output[max_idx] = T(1);
}


float_type reveal() const{
    return FloatFixedConverter<float, INT_TYPE, UINT_TYPE, FRACTIONAL_VALUE>::ufixed_to_float(this->s1);
}

/* void truncate(){ */
/*     this->s1 = truncate(this->s1); */
/* } */

};
