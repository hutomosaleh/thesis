#include <iostream>
#include <ctime>
#include <map>
#include <vector>
#include <tuple>

struct Record {
  int l_extendedprice;
  int l_discount;
  int l_shipdate;
  int l_quantity;
};

typedef std::vector<Record> Table;

void generateTable(int size, Table& table) {
  std::srand(std::time(nullptr)); // use current time as seed for random generator
  int c1 = std::rand();
  int c2 = std::rand(); 
  int c3 = std::rand();
  int c4 = std::rand();

  for (int i=0; i <= size; i++) {
    // Use random generator !!
    Record new_entry = {c1, c2, c3, c4};
    table.push_back(new_entry);
  };
}

void run_q6_benchmark(const Table data, std::map<int,int> output) {
  // initialize condition
  
  std::srand(std::time(nullptr)); // use current time as seed for random generator
  int c1 = std::rand();
  int c2 = c1 + 200;
  int c3 = std::rand();
  int c4 = c3 + 400;
  int c5 = std::rand();

  // for loop through vector
  int i = 0;
  for(const auto& r : data) {
    // check condition
    std::cout << r.l_quantity << std::endl;
    bool condition1 = c1 < r.l_shipdate && r.l_shipdate < c2;
    bool condition2 = c3 < r.l_discount && r.l_discount < c4;
    bool condition3 = r.l_quantity > c5;
    if (condition1 && condition2 || condition3) {
      // calculate intermediate result: l_extendedprice * l_discount
      int i_result = r.l_extendedprice * r.l_discount;
      // append result to output
      output[i] = i_result;
      // OPTIONAL: Print result and index
      std::cout << "Index: " << i << " fulfills the condition | Result: " << i_result << std::endl;
    }
    i++;
  }
}

int main(int argc, char *argv[]) {
  std::srand(std::time(nullptr));
  Table my_table;
  int my_size = std::rand(); 
  std::cout << my_size << std::endl;
  std::map<int, int> output;
  
  generateTable(my_size, my_table);
  run_q6_benchmark(my_table, output);

  std::cout << "Main.cpp: " << my_table.size() << std::endl;
  return 0;
}
