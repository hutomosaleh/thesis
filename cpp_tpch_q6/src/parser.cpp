#include <fstream>
#include <iostream>
#include <sstream>

#include "parser.hpp"

#define QUANTITY 4
#define EXTENDED_PRICE 5
#define DISCOUNT 6
#define SHIPDATE 10
#define NUM_COLUMN 16
#define DELIMITER '|'

template<typename T>
void bin2ptr(const char* filename, T **ptr)
{
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);

    // get its size:
    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // read the data
    char *temp = new char[fileSize];
    file.read(temp, fileSize);
    *ptr = reinterpret_cast<T*>(temp);
}

int Parser::dtoi(std::string str) {
  std::istringstream date(str);
  std::string time;
  int result = 0;
  int count = 0;
  while (getline(date, time, '-')) {
    int multiplier = 1;
    if (count==0) multiplier=365;
    if (count==1) multiplier=30;
    ++count;
    result += stoi(time)*multiplier;
  }
  return result;
}

void Parser::parse(std::string path, LineItem& record, bool overwrite_file)
{ 
  std::cout << "Generating TPCH Data" << std::endl;
  bool file_exists = true;
  if (overwrite_file)
  {
    std::ofstream l_quantity ("data/l_quantity.bin", std::ios::binary | std::ios::app);
    std::ofstream l_extendedprice ("data/l_extendedprice.bin", std::ios::binary | std::ios::app);
    std::ofstream l_discount ("data/l_discount.bin", std::ios::binary | std::ios::app);
    std::ofstream l_shipdate ("data/l_shipdate.bin", std::ios::binary | std::ios::app);
    std::ofstream size ("data/size.bin", std::ios::binary | std::ios::app);
    std::cout << "Parsing lineitem" << std::endl;

    std::fstream buffer(path);
    std::string line;
    int record_size = 0;
    while (getline(buffer, line)) {
      std::istringstream row(line);
      std::string field;
      int column = 0;
      while (getline(row, field, DELIMITER)) {
        if (column==QUANTITY) {
          double q = std::stod(field);
          l_quantity.write(reinterpret_cast<char*>(&q), sizeof(double));
        } else if (column==EXTENDED_PRICE) {
          double q = std::stod(field);
          l_extendedprice.write(reinterpret_cast<char*>(&q), sizeof(double));
        } else if (column==DISCOUNT) {
          double q = std::stod(field);
          l_discount.write(reinterpret_cast<char*>(&q), sizeof(double));
        } else if (column==SHIPDATE) {
          int q = dtoi(field);
          l_shipdate.write(reinterpret_cast<char*>(&q), sizeof(int));
        }
        ++column;
        if (column==NUM_COLUMN) {
          column = 0;
          continue;
        }
      }
      ++record_size;
    }
    size.write(reinterpret_cast<char*>(&record_size), sizeof(int));
  }
  else
  {
    std::cout << "Table already parsed in binary, using that instead." << std::endl;
  }

  // Write binary to variables
  bin2ptr("data/l_quantity.bin", &record.l_quantity);
  bin2ptr("data/l_extendedprice.bin", &record.l_extendedprice);
  bin2ptr("data/l_discount.bin", &record.l_discount);
  bin2ptr("data/l_shipdate.bin", &record.l_shipdate);
  bin2ptr("data/size.bin", &record.size);
}

