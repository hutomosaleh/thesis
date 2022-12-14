#include "cpu_kernels.hpp"

#include "defs.hpp"

void check_cpu(int n, double* l_quantity, int* l_shipdate, double* l_discount)
{
  for (int i = 0; i < n; i++) {
    bool valid_date = (l_shipdate[i] >= DATE_BOTTOM_LIMIT && l_shipdate[i] <= DATE_UPPER_LIMIT);
    bool valid_quantity = (l_quantity[i] < QUANTITY_LIMIT);
    bool valid_discount = (l_discount[i] >= DISCOUNT_BOTTOM_LIMIT && l_discount[i] < DISCOUNT_UPPER_LIMIT);
    l_quantity[i] = (valid_date && valid_quantity && valid_discount) ? 1 : 0;
  }
}

void multiply_cpu(int n, double* l_quantity, double* l_extendedprice, double* l_discount)
{
  for (int i = 0; i < n; i++) {
    l_extendedprice[i] = (l_quantity[i]) ? l_extendedprice[i]*l_discount[i] : 0;
  }
}
